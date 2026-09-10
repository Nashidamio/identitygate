import hashlib
import inspect
import textwrap
import types

import torch


_PTR_ROWS = {}
_STATS = {}


def reset_filter_stats():
    _PTR_ROWS.clear()
    _STATS.clear()
    _STATS.update({
        "spatial_blocks_seen": 0,
        "spatial_blocks_with_mask": 0,
        "spatial_masked_tokens": 0,
        "pointer_groups_seen": 0,
        "pointer_groups_with_mask": 0,
        "pointer_masked_tokens": 0,
    })


def get_filter_stats():
    return dict(_STATS)


def mark_output_block_rows(output, block_rows):
    rows = [bool(x) for x in block_rows]
    output["_identitygate_block_rows"] = rows


def _rows_tensor(rows, batch_size, device):
    if rows is None:
        return torch.zeros(
            batch_size,
            device=device,
            dtype=torch.bool,
        )

    out = torch.as_tensor(
        rows,
        device=device,
        dtype=torch.bool,
    ).flatten()

    if out.numel() != batch_size:
        raise RuntimeError(
            "IdentityGate row-mask size {} != batch size {}".format(
                out.numel(),
                batch_size,
            )
        )

    return out


def _identitygate_spatial_mask(prev, batch_size, seq_len, device):
    rows = _rows_tensor(
        prev.get("_identitygate_block_rows", None),
        batch_size,
        device,
    )

    mask = rows[:, None].expand(
        batch_size,
        seq_len,
    ).clone()

    _STATS["spatial_blocks_seen"] += 1

    if bool(rows.any().item()):
        _STATS["spatial_blocks_with_mask"] += 1
        _STATS["spatial_masked_tokens"] += int(mask.sum().item())

    return mask


def _identitygate_tag_ptr(out):
    ptr = out["obj_ptr"]

    _PTR_ROWS[id(ptr)] = out.get(
        "_identitygate_block_rows",
        None,
    )

    return ptr


def _identitygate_pointer_mask(
    pos_and_ptrs,
    batch_size,
    num_obj_ptr_tokens,
    device,
):
    n_ptrs = len(pos_and_ptrs)

    if n_ptrs <= 0:
        raise RuntimeError(
            "Pointer-mask helper called with zero pointers"
        )

    if num_obj_ptr_tokens % n_ptrs != 0:
        raise RuntimeError(
            "Pointer token count {} not divisible by pointer count {}".format(
                num_obj_ptr_tokens,
                n_ptrs,
            )
        )

    tokens_per_ptr = num_obj_ptr_tokens // n_ptrs

    base = torch.zeros(
        batch_size,
        n_ptrs,
        device=device,
        dtype=torch.bool,
    )

    for j, item in enumerate(pos_and_ptrs):
        ptr = item[1]
        rows = _rows_tensor(
            _PTR_ROWS.get(id(ptr), None),
            batch_size,
            device,
        )
        base[:, j] = rows

    mask = base.repeat_interleave(
        tokens_per_ptr,
        dim=1,
    )

    _STATS["pointer_groups_seen"] += n_ptrs

    if bool(base.any().item()):
        _STATS["pointer_groups_with_mask"] += int(
            base.any(dim=0).sum().item()
        )
        _STATS["pointer_masked_tokens"] += int(mask.sum().item())

    return mask


def install_identitygate_attention_filter(predictor):
    bound = getattr(
        predictor,
        "_prepare_memory_conditioned_features",
        None,
    )

    if bound is None or not hasattr(bound, "__func__"):
        raise RuntimeError(
            "Cannot locate _prepare_memory_conditioned_features"
        )

    original = bound.__func__
    source = textwrap.dedent(
        inspect.getsource(original)
    )

    original_sha256 = hashlib.sha256(
        source.encode("utf-8")
    ).hexdigest()

    spatial_old = (
        "torch.zeros(B, seq_len, device=device, dtype=bool)"
    )
    spatial_new = (
        "_identitygate_spatial_mask(prev, B, seq_len, device)"
    )

    if source.count(spatial_old) != 1:
        raise RuntimeError(
            "Unexpected spatial-mask source pattern count: {}".format(
                source.count(spatial_old)
            )
        )

    patched = source.replace(
        spatial_old,
        spatial_new,
        1,
    )

    ptr_expr = "out[\"obj_ptr\"]"
    ptr_count = patched.count(ptr_expr)

    if ptr_count != 2:
        raise RuntimeError(
            "Unexpected obj_ptr source pattern count: {}".format(
                ptr_count
            )
        )

    patched = patched.replace(
        ptr_expr,
        "_identitygate_tag_ptr(out)",
    )

    ptr_mask_old = (
        "to_cat_prompt_mask.append(None)  "
        "# \"to_cat_prompt_mask\" is not used"
    )
    ptr_mask_new = (
        "to_cat_prompt_mask.append("
        "_identitygate_pointer_mask("
        "pos_and_ptrs, B, obj_ptrs.shape[0], device"
        "))"
    )

    if patched.count(ptr_mask_old) != 1:
        raise RuntimeError(
            "Unexpected pointer-mask source pattern count: {}".format(
                patched.count(ptr_mask_old)
            )
        )

    patched = patched.replace(
        ptr_mask_old,
        ptr_mask_new,
        1,
    )

    final_old = (
        "prompt_mask = None  "
        "# For now, we always masks are zeros anyways"
    )
    final_new = (
        "prompt_mask = torch.cat(to_cat_prompt_mask, dim=1)\n"
        "        if not bool(prompt_mask.any().item()):\n"
        "            prompt_mask = None"
    )

    if patched.count(final_old) != 1:
        raise RuntimeError(
            "Unexpected final-mask source pattern count: {}".format(
                patched.count(final_old)
            )
        )

    patched = patched.replace(
        final_old,
        final_new,
        1,
    )

    namespace = dict(original.__globals__)
    namespace.update({
        "_identitygate_spatial_mask": _identitygate_spatial_mask,
        "_identitygate_tag_ptr": _identitygate_tag_ptr,
        "_identitygate_pointer_mask": _identitygate_pointer_mask,
    })

    exec(
        compile(
            patched,
            original.__code__.co_filename,
            "exec",
        ),
        namespace,
    )

    replacement = namespace[original.__name__]

    predictor._prepare_memory_conditioned_features = types.MethodType(
        replacement,
        predictor,
    )

    patched_sha256 = hashlib.sha256(
        patched.encode("utf-8")
    ).hexdigest()

    reset_filter_stats()

    return {
        "method": original.__name__,
        "original_source_sha256": original_sha256,
        "patched_source_sha256": patched_sha256,
        "external_source_modified": False,
    }
