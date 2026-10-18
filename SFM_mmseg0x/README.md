# SFM Line 1（FCN-style qhull-MSAU）— mmseg 0.x 加载与评测代码

本目录包含论文 *Spatial Frequency Modulation for Semantic Segmentation*（TPAMI 2025）
**路线①（FCN-style 模型：ARS + qhull 非均匀上采样 + 级联 LPRM）** 全部 18 个
Cityscapes / PascalContext-59 模型的加载与评测代码，以及与 released checkpoint
一一对应的配置文件。

代码基于 [MMSegmentation 0.25.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.25.0)
（Apache-2.0），叠加论文的自定义模块（SS3PR 系列）。相对训练 fork 的差异：

- 只保留本路线需要的文件（上游 0.25.0 + 19 个 fork 文件覆盖）；
- `models/backbones/resnet.py` 截断至 `ResNetV1cSaliencySampler`，
  移除了训练树中其他实验的 backbone 类（NyResNet、ResNetFreqMix 等）；
- 清理了残留的调试 print 与私有路径；`tools/test.py` 移除了 fork 内部包 `mmcv_custom` 的导入；
- `__init__.py` 追加本路线的注册导入；18 份配置仅将数据路径中性化为 `./data/`。

除上述清理外，可执行代码与训练/评测所用的 fork 一致；清理后的代码在服务器上复验
18/18 配置建模 + 18/18 checkpoint 严格加载通过（见文末验证记录）。

## 环境安装

```bash
conda create -n sfm0x python=3.7 -y
conda activate sfm0x
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
pip install mmsegmentation==0.25.0 opencv-python scipy "cython<3" torch-dct tensorboard tqdm
# 编译 qhull 扩展（SaliencySampler.reverse 的位精确后端，scipy 同源源码）：
cd mmseg/models/backbones/spatial && python setup.py build_ext --inplace && cd -
```

> 训练/验证环境为 py3.7.11 + torch 1.7.1+cu110 + mmcv-full 1.5.3 + numpy 1.21.6；
> 本目录代码已在 py3.7.11 + torch 1.11.0+cu113 + numpy 1.21.6 + mmcv-full 1.5.3（Linux）
> 上完成验证。qhull 编译用到 numpy<1.26 与 scipy<1.10 的构建 API（py3.7 默认满足）；
> 更高版本的 Python/scipy 需自行适配 `spatial/setup.py`。

## 数据准备

```bash
cd <本目录>       # 评测命令均在本目录下执行
mkdir -p data
# Cityscapes：data/cityscapes/{leftImg8bit,gtFine}
#   官方 gtFine 只含 labelIds 标注；本包评测读 *_gtFine_labelTrainIds.png，需先用
#   上游 mmsegmentation 0.25.0 的 tools/convert_datasets/cityscapes.py
#   （pip install cityscapesscripts）生成 labelTrainIds。
# PascalContext-59：data/VOCdevkit/VOC2010；并按上游
#   tools/convert_datasets/pascal_context.py 的说明额外下载 Details Release 的
#   annotations_holistic，生成 SegmentationClassContext/ 与
#   ImageSets/SegmentationContext/{train,val}.txt（详见各配置 data_root）。
```

> 说明：`pip install mmsegmentation==0.25.0` 主要用于自动带入 matplotlib、prettytable、
> packaging 等依赖；运行评测**必须**设 `PYTHONPATH` 指向本目录，让本地的 `mmseg/`
> （含 SS3PR 自定义模块）覆盖 pip 版——漏设会报 `KeyError: 'ResNetV1cSaliencySampler'`。

## 模型库（checkpoint 见 [HuggingFace Linwei-Chen/SFM](https://huggingface.co/Linwei-Chen/SFM/tree/main/line1)）

| 配置 (`configs/`) | 模型 | 论文 mIoU | 训练日志实测 |
| --- | --- | --- | --- |
| `ccnet/ccnet_r50SS3PR-d32_cityscapes.py` | CCNet-R50 D32 | 76.3 | 76.34 |
| `ccnet/ccnet_r50SS3PR-d16_cityscapes.py` | CCNet-R50 D16 | 78.5 | 78.45 |
| `ccnet/ccnet_r50SS3PR-d8_cityscapes.py` | CCNet-R50 D8 | 79.3 | 79.39 |
| `ocnet/isanet_r50SS3PR-d32_cityscapes.py` | OCNet-R50 D32 | 75.8 | 75.82 |
| `ocnet/isanet_r50SS3PR-d16_cityscapes.py` | OCNet-R50 D16 | 78.0 | 78.00 |
| `ocnet/isanet_r50SS3PR-d8_cityscapes.py` | OCNet-R50 D8 | 79.7 | 79.74 |
| `fcn/FCN_r50SS3PR-d32_cityscapes.py` | FCN-R50 D32 | 76.2 | 76.20 |
| `fcn/FCN_r50SS3PR-d16_cityscapes.py` | FCN-R50 D16 | 77.8 | 77.78 |
| `fcn/FCN_r50SS3PR-d8_cityscapes.py` | FCN-R50 D8 | 78.7 | 78.65 |
| `pspnet/pspnet_r50SS3PR-d32_cityscapes.py` | PSPNet-R50 D32 | 76.0 | 76.18 |
| `pspnet/pspnet_r50SS3PR-d16_cityscapes.py` | PSPNet-R50 D16 | 78.1 | 78.05 |
| `pspnet/pspnet_r50SS3PR-d8_cityscapes.py` | PSPNet-R50 D8 | 79.5 | 79.52 |
| `pcaa/pcaa_r50SSPR-d32_cityscapes.py` | PCAA-R50（第一列） | 76.8 | 76.83 |
| `pcaa/pcaa_r50SSPR-d16_cityscapes.py` | PCAA-R50（第二列） | 77.9 | 77.86 |
| `fcn/FCN_r50SS3PR-d32_pascalcontext.py` | FCN-R50 D32 | 46.0 | 46.35 |
| `pspnet/pspnet_r50SS3PR-d32_pascalcontext.py` | PSPNet-R50 D32 | 49.7 | 49.68 |
| `ccnet/ccnet_r50SS3PR-d32_pascalcontext.py` | CCNet-R50 D32 | 49.1 | 49.07 |
| `ocnet/isanet_r50SS3PR-d32_pascalcontext.py` | OCNet-R50 D32 | 49.9 | 49.85 |

所有 released checkpoint 均经训练日志核验：实测值高于论文报告值，或与论文值在一位小数舍入后一致（如 PSPNet-R50 D16 实测 78.05，舍入 = 论文 78.1）。

## 评测

```bash
PYTHONPATH=<本目录> python tools/test.py \
    configs/ccnet/ccnet_r50SS3PR-d16_cityscapes.py \
    <下载的 iter_40000.pth> --eval mIoU
```

单卡跑多卡训练的 checkpoint 无需额外处理（`SamplesPerGPU` 已写入配置）。
评测指标打印在终端，并写入 `./work_dirs/<配置名>/eval_single_scale_*.json`。

## 验证记录（2026-10-16）

- 精简树在 py3.7.11 + torch 1.11.0+cu113 + numpy 1.21.6 + mmcv-full 1.5.3 上：
  **18/18 配置构建成功 + 18/18 checkpoint `load_state_dict(strict=True)` 全部通过**；
- 打包版（含数据路径中性化的 18 份配置）再次 **18/18 构建通过**；
- 端到端评测等价性此前已单独验证：CCNet-R50 D16 用真实 checkpoint 完整评测 500 张，
  qhull-MSAU 移植实现与原训练代码逐类 IoU 完全一致（见主仓库 README）。
