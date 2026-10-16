# TPAMI 2025 Spatial Frequency Modulation for Semantic Segmentation

[TPAMI 2025] Spatial Frequency Modulation for Semantic Segmentation

[Arxiv](https://arxiv.org/abs/2507.11893) [IEEE](https://ieeexplore.ieee.org/document/11095993)

![image-20250715191310237](README.assets/image-20250715191310237.png)

In this work, we identify and address the "aliasing degradation" problem in modern deep neural networks, where high-frequency information crucial for semantic segmentation is distorted during downsampling.

Instead of simply filtering out these valuable details, we introduce **Spatial Frequency Modulation (SFM)**, a novel framework that:

1. **Modulates** high-frequency features to a lower frequency band *before* downsampling, protecting them from aliasing.
2. **Demodulates** these features back to their original high frequency during upsampling, recovering fine-grained details for a more accurate segmentation.

Our lightweight and plug-and-play modules, **Adaptive Resampling (ARS)** and **Multi-Scale Adaptive Upsampling (MSAU)**, can be seamlessly integrated into various CNN and Transformer architectures to significantly boost their performance.

![image-20250715191346558](README.assets/image-20250715191346558.png)

> **Figure:** An illustration of our SFM framework. Adaptive Resampling (ARS) is inserted before downsampling layers to perform frequency modulation, and Multi-Scale Adaptive Upsampling (MSAU) is used to demodulate the features and produce the final high-resolution segmentation map.

## Main Contributions

- **Identifying "Aliasing Degradation":** We quantitatively demonstrate that a higher aliasing ratio in feature maps leads to lower segmentation accuracy, providing a clear motivation for frequency-aware network design.
- **Spatial Frequency Modulation (SFM):** We propose a novel framework to preserve high-frequency details by modulating them to lower frequencies to survive downsampling, and then demodulating them to recover the details.
- **Lightweight and Effective Modules:** We implement SFM with two novel modules:
  - **Adaptive Resampling (ARS):** A lightweight module that learns to densely sample high-frequency regions (e.g., boundaries, textures) to effectively lower their frequency representation.
  - **Multi-Scale Adaptive Upsampling (MSAU):** A module that performs non-uniform upsampling to reverse the modulation and refines segmentation details by modeling multi-scale pixel relationships.
- **Broad Applicability:** Our method is model-agnostic and consistently improves various state-of-the-art segmentation models, including CNN-based (ResNet, ConvNeXt, InternImage) and Transformer-based (Swin) architectures, with minimal computational overhead.

## Code & Model

Two release lines (as described in the paper):

**Line 1 — FCN-style models: ARS + MSAU (qhull non-uniform upsampling + cascaded LPRM).**
For non-FPN segmentation models (FCN / PSPNet / CCNet / OCNet / PCAA), MSAU first upsamples
the prediction **non-uniformly** via Delaunay-triangulation barycentric interpolation
(paper Eq. 17/18, "triangulation"), then refines it with a cascade of Local Pixel Relation
Modules (LPRM, dilations doubling 1→64).

**Line 2 — FPN models (e.g. Mask2Former): ARS + CARAFE-style reassembly + 3-point dynamic sampling weighted alignment.**

| Line | Model | Dataset | mIoU | Code / Config / Ckpt |
| --- | --- | --- | --- | --- |
| 1 (FCN) | CCNet-R50, D16 (SFM) | Cityscapes | 73.3 → **78.5** | ckpt (HF 待传)；训练 log 78.45，qhull-MSAU 移植版端到端实测逐类 IoU 与原代码完全一致 |
| 1 (FCN) | CCNet / OCNet / FCN / PSPNet-R50, D32/D16/D8 (SFM) | Cityscapes | 76.3/78.5/79.3 · 75.8/78.0/79.7 · 76.2/77.8/78.7 · 76.0/78.1/79.5 | ckpts (HF 上传中)，训练 log 逐一核验 ≥ 论文值 |
| 1 (FCN) | PCAA-R50 (SFM) | Cityscapes | 73.3 → 76.8 / 76.9 → 77.9 | ckpts (HF 上传中) |
| 1 (FCN) | FCN / PSPNet / CCNet / OCNet-R50, D32 (SFM) | PascalContext-59 | 43.8 → 46.0 / 48.0 → 49.7 / 47.7 → 49.1 / 47.9 → 49.9 | ckpts (HF 上传中) |
| 2 (FPN) | Mask2Former Swin-T (SFMv2) | ADE20K | 47.4 → **49.1** | [code](./SFM_mask2former/), [config](./SFM_mask2former/configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512_SFMv2.py), ckpt (HF 待传)；2025-10 复测 **49.34** |

> All released checkpoints are training-log verified to match or exceed the paper-reported numbers; runs whose best score falls below the paper value are not released.



## Quick Start: Plug-and-Play Usage (pip installable)

The core operators are packaged as a pip-installable package [`sfm/`](./sfm/) (pure PyTorch, no framework required):

```bash
pip install .            # from repo root, or:
pip install git+https://github.com/Linwei-Chen/SFM.git
```

It provides four plug-and-play pieces:

| Piece | Import | Replaces |
| --- | --- | --- |
| Adaptive Resampling (ARS, backbone downsampling) | `sfm.ARS` / `sfm.SaliencySampler` | the max-pool / stride-2 stage entry in your backbone |
| MSAU non-uniform upsampling for FCN-style models (Line 1) | `sfm.Interp2D` / `sfm.reverse_feat` (qhull Delaunay triangulation) | the bilinear `resize` before the LPRM cascade in FCN/PSPNet/CCNet-style decode paths |
| Aligned upsampling (FPN top-down, Line 2) | `sfm.LPRMAlignUpModule` | `resize(laterals[i])` in any FPN-like fusion |
| Ready-made FPN neck (Line 2) | `sfm.SFMFPN` | `neck=dict(type='FPN', ...)` |

**Way 1: seamless integration into MMDetection / MMSegmentation** — importing `sfm` auto-registers `SFMFPN` into the framework's neck registry (mmdet 2.x NECKS / 3.x MODELS, mmseg 0.x NECKS / 1.x MODELS; verified against pip mmdet 2.28.2 on GPU). Two config lines are all you need, see [`examples/`](./examples/):

```python
custom_imports = dict(imports=['sfm'], allow_failed_imports=False)
model = dict(neck=dict(type='SFMFPN', in_channels=[256, 512, 1024, 2048],
                       out_channels=256, num_outs=5, sfm_compress_ratio=16))
```

**Way 2: pure PyTorch (no framework)**

```python
from sfm import ARS, LPRMAlignUpModule, AlignModule, compute_real_grids

# FPN top-down fusion (the only change vs. official FPN):
align_up = LPRMAlignUpModule(256, 256, kernel_size=3, compress_ratio=16)
laterals[i - 1] = laterals[i - 1] + align_up(x_low=laterals[i],
                                             guidance_high_aligned=laterals[i - 1])

# ARS before a backbone stage (modulate high freq, then resample):
ss = ARS(feat_dim=256, stride=1, padding_size=5, fwhm=3,
         sampler_mode='avg_lhpf_psp', psp_ratio=16, align_corners=False)
x, x_sampled, saliency, grid = ss(x, sample_with_grid=True)  # grid = warp record

# If your backbone uses ARS, flatten warped features before FPN/decoder:
real_grids = compute_real_grids(grids, feats[0].shape, align_corners=False)
feat = AlignModule(channels, k=3)(feat, real_grids[level_idx])  # non-uniform → uniform
```

The minimal `ARS` is **numerically identical to the original training code** (state_dict key-for-key compatible; outputs bit-exact on all three training trees, with `psp_bins` selecting the Mask2Former line `[1,2,3,6]` vs. the UPerNet/InternImage line `[1,2,3,7]`). Its SyncBN works out-of-the-box under DDP training; for single-GPU/inference call `sfm.swap_syncbn_to_bn(model)` first.

**Way 3: qhull non-uniform upsampling (Line 1, FCN-style models)**

In the FCN line, the warped (non-uniform) feature/prediction is flattened back by
Delaunay-triangulation barycentric interpolation (paper Eq. 17/18) — `SaliencySampler.reverse`
in the original code. The package provides it as `sfm.interpolate`:

```python
from sfm import ARS, reverse_feat
from sfm.interpolate import Interp2D, qhull_backend

qhull_backend()          # report/select backend: vendored compiled qhull → scipy.spatial.Delaunay fallback

ss = ARS(feat_dim=64, sampler_mode='psp', psp_bins=[1, 2, 4, 8])   # ARS warps the grid
x, x_sampled, saliency, grid = ss(x)

# ... downsampling stages run on x_sampled ...

# flatten back to the uniform grid (bit-exact port of the original SaliencySampler.reverse):
reversed_feat = ss.reverse(sampled_feat, grid, h, w)      # h, w = original feature size
# equivalently, function form (also caches a reusable inverse sampler):
reversed_feat, inv_sampler = reverse_feat(sampled_feat, grid, h, w,
                                          align_corners=True, reverse_sampler=None)
# batched re-flattening with a cached inverse sampler:
reversed_feat = Interp2D(size=(h, w), align_corners=True)(sampled_feat, inv_sampler)
```

`Interp2D`/`reverse_feat` are **bit-exact against the original training code** (verified
module-level on random grids and on real ARS-produced grids, and end-to-end on Cityscapes:
the ported implementation with the released CCNet checkpoint reproduces the original code's
per-class IoU exactly — details in `docs/`). The qhull provider is resolved in order:
1. `sfm.vendor.spatial.qhull` — scipy's qhull Cython source compiled by `pip install .`
   (needs a C compiler + LAPACK; LAPACK is auto-detected, falling back to BLAS bundled in
   numpy/scipy wheels). This is the bit-exact-as-scipy path with `find_simplex(..., return_c=True)`.
2. a top-level `qhull` package, if present.
3. `scipy.spatial.Delaunay` — same Qhull engine, verified bit-exact on real ARS grids
   (differs only for pathological outside-point cases).

Run the demos and tests:

```bash
python examples/pure_torch_sfm_fpn.py   # end-to-end pure-torch demo
python tests/test_sfm_package.py        # 19 unit tests (ARS / FPN)
python tests/test_sfm_align.py          # unit tests (CARAFE-style reassembly / aligned upsampling)
```

Single-file alternative (frozen copy, for copy-paste into other codebases): [`SFM_FPN.py`](./SFM_FPN.py).

## Citation

If you find our work useful in your research, please consider citing our paper:

Generated bibtex

```
@article{chen2023spatialfrequency,
  title={Spatial Frequency Modulation for Semantic Segmentation},
  author={Chen, Linwei and Fu, Ying and Gu, Lin and Zheng, Dezhi and Dai, Jifeng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  volume={47},
  number={11},
  pages={9767-9784},
  doi={10.1109/TPAMI.2025.3592621}
}
    
```

## Acknowledgements

This project is built upon the excellent [MMSegmentation](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fopen-mmlab%2Fmmsegmentation) toolbox. We thank the authors for their open-source contribution.

## Contact

If you encounter any problems or bugs, please don't hesitate to contact me at [chenlinwei@bit.edu.cn](chenlinwei@bit.edu.cn), [charleschen2013@163.com](charleschen2013@163.com). To ensure effective assistance, please provide a brief self-introduction, including your name, affiliation, and position. If you would like more in-depth help, feel free to provide additional information such as your personal website link. I would be happy to discuss with you and offer support.

