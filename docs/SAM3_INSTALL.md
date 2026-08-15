# SAM 3 install record

- Repo: https://github.com/facebookresearch/sam3
- Commit: 8f0b7f4d4e7eda2ed606ebde6702c93359ad01da
- Installed editable: `pip install -e .` from `~/thesis/externals/sam3`
- Verified: build_sam3_image_model, build_sam3_video_predictor, build_sam3_multiplex_video_predictor all importable

## Required pins beyond SAM 3's declared deps
| Package        | Version | Why                                                                     |
|----------------|---------|-------------------------------------------------------------------------|
| setuptools     | <82     | SAM 3 model_builder.py imports pkg_resources, removed in setuptools 82  |
| einops         | any     | SAM 3 sam/rope.py imports einops (declared only in [notebooks] extra)   |
| pycocotools    | any     | SAM 3 imports pycocotools at load time (declared only in [dev] extra)   |

These are upstream SAM 3 bugs (missing runtime deps). Reinstall order after any env rebuild:
1. torch/torchvision (cu128 wheels)
2. `pip install -e ~/thesis/externals/sam3`
3. `pip install "setuptools<82" einops pycocotools`
4. Re-run the import verification below.

## Import verification
```python
import sam3
from sam3.model_builder import (
    build_sam3_image_model,
    build_sam3_video_predictor,
    build_sam3_multiplex_video_predictor,
)
```
Expected warnings (harmless): pkg_resources deprecation, timm.models.layers deprecation.
