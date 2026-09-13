import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/EXP036-ws-keyboard-adjudication-v1.json"


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_clean_committed_tree():
    required = [
        "configs/EXP036-ws-keyboard-adjudication-v1.json",
        "scripts/exp036_ws_keyboard_adjudicator.py",
    ]

    for rel in required:
        subprocess.check_call(
            ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        text=True,
    ).strip()

    if status:
        raise RuntimeError("EXP036 requires a clean committed tree")


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "exp035_review_pack",
        str(path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import EXP035 visual functions")

    module = importlib.util.module_from_spec(spec)
    sys.modules["exp035_review_pack"] = module
    spec.loader.exec_module(module)
    return module


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: r["scene_event_id"])


def atomic_write_labels(path, rows, labels):
    tmp = path.with_suffix(".tmp")

    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_event_id",
                "manual_label",
                "reason",
                "reviewer_notes",
            ],
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            sid = row["scene_event_id"]
            value = labels.get(
                sid,
                {
                    "manual_label":"",
                    "reason":"",
                    "reviewer_notes":"",
                },
            )

            writer.writerow(
                {
                    "scene_event_id":sid,
                    "manual_label":value["manual_label"],
                    "reason":value["reason"],
                    "reviewer_notes":value["reviewer_notes"],
                }
            )

    tmp.replace(path)


def load_existing_labels(path):
    if not path.exists():
        return {}

    out = {}

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["scene_event_id"]] = {
                "manual_label":row["manual_label"],
                "reason":row["reason"],
                "reviewer_notes":row["reviewer_notes"],
            }

    return out


class ReviewState:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rows = load_rows(ROOT / cfg["source_csv"])

        if len(self.rows) != int(cfg["expected_scenes"]):
            raise RuntimeError("Unexpected disagreement scene count")

        source = ROOT / cfg["source_csv"]

        if sha256sum(source) != cfg["source_csv_sha256"]:
            raise RuntimeError("Source disagreement CSV SHA mismatch")

        exp035_cfg = ROOT / cfg["exp035_config"]

        if sha256sum(exp035_cfg) != cfg["exp035_config_sha256"]:
            raise RuntimeError("EXP035 config SHA mismatch")

        self.exp035_cfg = json.loads(
            exp035_cfg.read_text(encoding="utf-8")
        )

        self.exp035 = load_module(
            ROOT / "scripts/exp035_make_ws_review_pack.py"
        )

        self.frame_root = Path(cfg["frame_root"])

        self.output_root = Path(
            cfg["output_root"]
        ).expanduser()

        self.cache_dir = self.output_root / cfg["cache_dir"]
        self.labels_path = self.output_root / cfg["labels_file"]

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.labels = load_existing_labels(self.labels_path)

        atomic_write_labels(
            self.labels_path,
            self.rows,
            self.labels,
        )

        self.keymap = {
            key: {
                "manual_label":value[0],
                "reason":value[1],
                "reviewer_notes":"",
            }
            for key, value in cfg["keys"].items()
        }

    def filled(self):
        return sum(
            bool(v["manual_label"].strip())
            for v in self.labels.values()
        )

    def first_unlabeled(self):
        for i, row in enumerate(self.rows):
            sid = row["scene_event_id"]
            if not self.labels.get(
                sid,
                {},
            ).get("manual_label", "").strip():
                return i
        return 0

    def next_unlabeled(self, start):
        n = len(self.rows)

        for step in range(1, n + 1):
            i = (start + step) % n
            sid = self.rows[i]["scene_event_id"]

            if not self.labels.get(
                sid,
                {},
            ).get("manual_label", "").strip():
                return i

        return start

    def save_label(self, index, key):
        if key not in self.keymap:
            raise ValueError("Invalid label key")

        row = self.rows[index]
        sid = row["scene_event_id"]

        self.labels[sid] = dict(self.keymap[key])

        atomic_write_labels(
            self.labels_path,
            self.rows,
            self.labels,
        )

        return self.next_unlabeled(index)

    def image_path(self, index):
        row = self.rows[index]
        sid = row["scene_event_id"]
        safe = sid.replace(":", "__")
        out = self.cache_dir / (safe + ".png")

        if out.exists():
            return out

        paths = self.exp035.get_frame_paths(
            self.frame_root,
            row["video"],
        )

        indices = self.exp035.selected_indices(
            row,
            len(paths),
            self.exp035_cfg,
        )

        width = int(
            self.exp035_cfg[
                "visual_context"
            ][
                "contact_sheet_thumbnail_width"
            ]
        )

        self.exp035.make_contact_sheet(
            sid,
            paths,
            indices,
            width,
            out,
        )

        return out

def html_page(state, index):
    row = state.rows[index]
    sid = row["scene_event_id"]

    current = state.labels.get(
        sid,
        {},
    ).get("manual_label", "")

    return """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>EXP036 WS Review</title>
<style>
body { font-family: sans-serif; background:#111; color:#eee; margin:20px; }
#top { position:sticky; top:0; background:#111; padding:8px; z-index:10; }
img { max-width:100%%; height:auto; display:block; margin:auto; }
.key { display:inline-block; margin:4px 10px 4px 0; padding:5px 8px; border:1px solid #777; }
</style>
</head>
<body>
<div id="top">
<h2>%s</h2>
<div>Scene %d / %d | completed %d / %d | current label: %s</div>
<div>
<span class="key">1 WHOLE_SCENE / cut-switch</span>
<span class="key">2 WHOLE_SCENE / global obstruction</span>
<span class="key">3 NORMAL / local event</span>
<span class="key">4 NORMAL / continuous camera motion</span>
<span class="key">0 skip</span>
<span class="key">left previous</span>
<span class="key">right next</span>
</div>
</div>

<img src="/image?i=%d">

<script>
const currentIndex = %d;
const total = %d;

document.addEventListener("keydown", async (e) => {
  if (["1","2","3","4"].includes(e.key)) {
    const r = await fetch("/label", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({index:currentIndex,key:e.key})
    });
    const x = await r.json();
    window.location = "/?i=" + x.next;
  } else if (e.key === "0" || e.key === "ArrowRight") {
    window.location = "/?i=" + ((currentIndex + 1) %% total);
  } else if (e.key === "ArrowLeft") {
    window.location = "/?i=" + ((currentIndex - 1 + total) %% total);
  }
});
</script>
</body>
</html>
""" % (
        sid,
        index + 1,
        len(state.rows),
        state.filled(),
        len(state.rows),
        current if current else "UNLABELED",
        index,
        index,
        len(state.rows),
    )


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def send_bytes(self, content, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            parsed = urlparse(self.path)
            q = parse_qs(parsed.query)

            if parsed.path == "/":
                if "i" in q:
                    index = int(q["i"][0])
                else:
                    index = state.first_unlabeled()

                index %= len(state.rows)
                self.send_bytes(html_page(state, index).encode("utf-8"), "text/html; charset=utf-8")
                return

            if parsed.path == "/image":
                index = int(q["i"][0]) % len(state.rows)
                path = state.image_path(index)
                self.send_bytes(path.read_bytes(), "image/png")
                return

            if parsed.path == "/status":
                payload = json.dumps({"total":len(state.rows),"filled":state.filled(),"labels_file":str(state.labels_path)}).encode("utf-8")
                self.send_bytes(payload, "application/json")
                return

            self.send_error(404)

        def do_POST(self):
            if self.path != "/label":
                self.send_error(404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            index = int(payload["index"])
            key = str(payload["key"])
            next_index = state.save_label(index, key)

            result = json.dumps({"ok":True,"next":next_index,"filled":state.filled()}).encode("utf-8")
            self.send_bytes(result, "application/json")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    require_clean_committed_tree()

    cfg = json.loads(
        CFG_PATH.read_text(encoding="utf-8")
    )

    state = ReviewState(cfg)

    host = cfg["host"]
    port = int(cfg["port"])

    print("EXP036_KEYBOARD_REVIEW_READY")
    print("URL=http://{}:{}".format(host, port))
    print("TOTAL={}".format(len(state.rows)))
    print("FILLED={}".format(state.filled()))
    print("LABELS={}".format(state.labels_path))
    print("Press Ctrl+C only when you want to stop the server.")

    server = ThreadingHTTPServer(
        (host, port),
        make_handler(state),
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
