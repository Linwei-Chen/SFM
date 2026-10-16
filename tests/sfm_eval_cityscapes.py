# -*- coding: utf-8 -*-
"""FCN 线（qhull-MSAU）真实 checkpoint 实测：Cityscapes CCNet-R50-D16 完整评测。

在不改动原始代码树的前提下，让 2022 年 py3.7 环境的训练配置在本机 py3.9 复跑：
  1. sys.modules 桩：DCNv3 CUDA 扩展（CCNet 用不到，防树 import 连带失败）、
     matplotlib、natten、mmseg.* 的 spatial 包（调试用逆插值不触达）；
  2. **mmseg_custom.models.backbones.spatial.qhull → 本包 vendored py3.9 编译版**
     （scipy 同源 Cython 源码，三角剖分逐位同源）——原 SaliencySampler.reverse
     的 qhull 非均匀上采样由此真实参与推理；
  3. R2 轮把原树 interp2d.Interp2D 替换为 sfm.Interp2D（移植版），
     与 R1（原版类 + 同一 qhull）逐类 IoU 对拍，验证移植版端到端等价。

用法（mmseg env，GPU 2）:
    cd /data4/chenlinwei/code/sfm_eval_dcn
    PYTHONPATH=/data4/chenlinwei/code/ResolutionDet/InternImage/segmentation \
    CUDA_VISIBLE_DEVICES=2 SMOKE=0 SFM_INJECT=0 \
    /data4/chenlinwei/anaconda3/envs/mmseg/bin/python sfm_eval_cityscapes.py
结果:
    R1 原版代码    → eval_cs_r1_orig.txt
    R2 sfm 注入版  → eval_cs_r2_sfm.txt
"""

import os
import sys
import types

# ---------------------------------------------------------------------------
# 1. 桩（必须在导入 mmseg_custom 之前生效）
# ---------------------------------------------------------------------------
_dcnv3 = types.ModuleType('DCNv3')


def _no_cuda(*a, **k):
    raise RuntimeError('DCNv3 CUDA 扩展桩：CCNet 评测不应触达')


_dcnv3.dcnv3_forward = _no_cuda
_dcnv3.dcnv3_backward = _no_cuda
sys.modules['DCNv3'] = _dcnv3

import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
print('[tf32] allow_tf32 = False')

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKDIR)


def _stub_pkg_getattr(name):
    if name.startswith('__'):
        raise AttributeError(name)
    raise RuntimeError(f'stub 模块属性被触达: {name}（评测不应使用）')


class _PermissiveDummy:
    """宽式哑元：属性访问静默通过，真正调用时才报错；可被继承（__mro_entries__）。"""

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return _PermissiveDummy()

    def __call__(self, *a, **k):
        raise RuntimeError('permissive 桩被调用（评测不应触达）')

    def __mro_entries__(self, bases):
        return (object,)

    def __iter__(self):
        raise RuntimeError('permissive 桩被迭代（评测不应触达）')

    def __getitem__(self, k):
        raise RuntimeError('permissive 桩被索引（评测不应触达）')


def _permissive_module(fullname):
    m = types.ModuleType(fullname)
    m.__path__ = []

    def _getattr(name):
        if name.startswith('__'):
            raise AttributeError(name)
        return _PermissiveDummy()

    m.__getattr__ = _getattr
    sys.modules[fullname] = m
    return m


def _stub_module(fullname, replace_with=None):
    m = replace_with if replace_with is not None else types.ModuleType(fullname)
    if replace_with is None:
        m.__path__ = []
        m.__getattr__ = _stub_pkg_getattr
    sys.modules[fullname] = m
    return m


# matplotlib/natten 等评测不触达的编译依赖（宽式桩：import 链放行，调用即炸）
_mpl = _permissive_module('matplotlib')
_mpl.pyplot = _permissive_module('matplotlib.pyplot')
_natten = _permissive_module('natten')
_natten_f = _permissive_module('natten.functional')
for _name in ('natten2dav', 'natten2dqkrpb'):
    setattr(_natten_f, _name, _no_cuda)

# vendored py3.9 qhull（先导入，作为 spatial.qhull 的替身）
import sfm  # noqa: E402  (WORKDIR 下的发布包)
from sfm import interpolate as _sfm_interp  # noqa: E402
import sfm.vendor.spatial.qhull as _vendored_qhull  # noqa: E402
print('[qhull] sfm backend =', _sfm_interp.qhull_backend())

# 可编辑 mmseg 树与 mmseg_custom 树的 spatial：父包桩掉（真实 __init__ 会拉
# py37 .so 兄弟模块），qhull 名字注入 vendored py3.9 编译版——两棵树的
# SaliencySampler.reverse 由此真实可用（CCNet 跑的是可编辑树的类）。
# 注意 `import a.b.c as x` 在 py3.9 会 getattr(父包, 'c') 绑定，因此父桩
# 必须带真实属性 qhull
_spatial_mmseg = _stub_module('mmseg.models.backbones.spatial')
_spatial_mmseg.qhull = _vendored_qhull
_stub_module('mmseg.models.backbones.spatial.qhull', _vendored_qhull)
_spatial_custom = _stub_module('mmseg_custom.models.backbones.spatial')
_spatial_custom.qhull = _vendored_qhull
_stub_module('mmseg_custom.models.backbones.spatial.qhull', _vendored_qhull)

# 可编辑树 interp2d.py 还 `from mmcv.ops.carafe import Tensor`（mmcv 1.5.0
# 无此名）：给真实 carafe 模块补属性即可，无需桩。mmcv.ops 是懒加载器，
# 必须用 importlib.import_module 拿真实模块对象
import importlib  # noqa: E402

_carafe_mod = importlib.import_module('mmcv.ops.carafe')
if not hasattr(_carafe_mod, 'Tensor'):
    _carafe_mod.Tensor = torch.Tensor

# 两棵树注册同名自定义模块 → 强制 force 注册
from mmcv.utils.registry import Registry  # noqa: E402

_orig_register_module = Registry.register_module


def _force_register(self, *args, **kwargs):
    kwargs['force'] = True
    return _orig_register_module(self, *args, **kwargs)


Registry.register_module = _force_register

# ---------------------------------------------------------------------------
# 2. 导入框架与原始代码树（PYTHONPATH 指向 InternImage/segmentation）
# ---------------------------------------------------------------------------
import mmcv  # noqa: E402
import mmseg  # noqa: E402,F401
import mmseg_custom  # noqa: E402  注册全部自定义模块
from mmcv.runner import load_checkpoint  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmseg.apis import single_gpu_test  # noqa: E402
from mmseg.datasets import build_dataloader, build_dataset  # noqa: E402
from mmseg.models import build_segmentor  # noqa: E402

# ---------------------------------------------------------------------------
# 3. R2 轮：原树 Interp2D → sfm 移植版（模块全局名替换，reverse 即刻生效）
# ---------------------------------------------------------------------------
SFM_INJECT = os.environ.get('SFM_INJECT', '0') == '1'
if SFM_INJECT:
    # 两棵树的 interp2d 都替换（CCNet 实际跑可编辑树的 SaliencySampler）
    import mmseg.models.backbones.interp2d as _orig_interp2d

    _orig_interp2d.Interp2D = _sfm_interp.Interp2D
    import mmseg_custom.models.backbones.interp2d as _orig_interp2d_custom

    _orig_interp2d_custom.Interp2D = _sfm_interp.Interp2D
    print('[inject] mmseg + mmseg_custom 的 interp2d.Interp2D -> sfm.Interp2D')
else:
    print('[baseline] 原版 interp2d.Interp2D（同一 vendored qhull）')

# 诊断钩子：Delaunay 失败时把点集存盘（DEBUG_DEL AUNAY=1 时启用）
if os.environ.get('DEBUG_DELAUNAY', '0') == '1':
    _orig_Delaunay = _vendored_qhull.Delaunay

    class _DebugDelaunay:
        def __new__(cls, points, *a, **k):
            try:
                return _orig_Delaunay(points, *a, **k)
            except Exception:
                import numpy as _np
                p = points.cpu().numpy() if hasattr(points, 'cpu') \
                    else _np.asarray(points)
                _np.save('/tmp/fail_points.npy', p)
                print(f'[debug] 失败点集已存 /tmp/fail_points.npy '
                      f'shape={p.shape} dtype={p.dtype}')
                raise

    _vendored_qhull.Delaunay = _DebugDelaunay
    print('[debug] Delaunay 失败截获钩子已启用')

# ---------------------------------------------------------------------------
# 3.5 原树推理 bug 的内存修正（不改树文件）
#
# 现树上 get_final_feat 的 reverse 调用把 `img.shape[-3], img.shape[-4]`
# （NCHW 下 = C=3, N=1）当作 h、w 传给 SaliencySampler.reverse——这是训练
# （2022-09-05, val mIoU 0.7845）之后被改坏的实验残留：qhull 会收到
# y∈[0,2]、x≡0 的退化点集（QH6154 flat simplex）。训练当时该行必然正确。
# 这里用 inspect+exec 在内存中重定义 get_final_feat，把 h/w 修正为
# img.shape[-2], img.shape[-1]（即 H, W），并绑定回 PixelRelation 类。
# ---------------------------------------------------------------------------
import inspect  # noqa: E402
import textwrap  # noqa: E402

_PixelRelation = mmseg_custom.models.segmentors.encoder_decoder.EncoderDecoderWithSaliencySamplerPixelRelation
_src = inspect.getsource(_PixelRelation.get_final_feat)
if 'img.shape[-3], img.shape[-4]' not in _src:
    raise RuntimeError('原树 get_final_feat 源码与预期不符，请人工检查')
_src_fixed = textwrap.dedent(_src).replace(
    'img.shape[-3], img.shape[-4]', 'img.shape[-2], img.shape[-1]')
_encdec_mod = sys.modules['mmseg_custom.models.segmentors.encoder_decoder']
exec(compile(_src_fixed, '<get_final_feat-fixed>', 'exec'),
     _encdec_mod.__dict__)
_PixelRelation.get_final_feat = _encdec_mod.__dict__['get_final_feat']
print('[fix] get_final_feat 的 reverse h/w: img.shape[-3],[-4](=C,N) '
      '-> img.shape[-2],[-1](=H,W)，原树文件未改动')

# ---------------------------------------------------------------------------
# 4. 配置：训练 config 原样加载 + 内存 patch（不改树）
# ---------------------------------------------------------------------------
TREE = '/data4/chenlinwei/code/ResolutionDet/InternImage/segmentation'
EXP = ('/data4/chenlinwei/code/ResolutionDet_Exp_data3/mmseg_exp/'
       'CCNet_Cityscapes/BS8_LR0.01_D16_SS3_100_none_d1-2-4-8-16-32-64_4xup_40k')
CFG = os.path.join(EXP, 'ccnet_r50SS3PR-d16_cityscapes.py')
CKPT = os.path.join(EXP, 'iter_40000.pth')
DATA_ROOT = '/data4/chenlinwei/dataset/'

cfg = mmcv.Config.fromfile(CFG)
if cfg.model.get('backbone') is not None and 'init_cfg' in cfg.model.backbone:
    cfg.model.backbone.init_cfg = None
cfg.model.pretrained = None
cfg.data.test.data_root = DATA_ROOT
cfg.data.workers_per_gpu = 4
cfg.gpu_ids = [0]
cfg.work_dir = WORKDIR
cfg.load_from = None

print(f'[cfg] segmentor={cfg.model.type}, test mode={cfg.model.test_cfg}')
print(f'[cfg] pixel_relation={cfg.model.get("pixel_relation")}, '
      f'reverse_final in config={cfg.model.get("reverse_final", "(未设→类默认 True)")}')
print('[cfg] data.test:', cfg.data.test.img_dir, '->', cfg.data.test.data_root)

# ---------------------------------------------------------------------------
# 5. 构建模型 + 加载真实 checkpoint
# ---------------------------------------------------------------------------
model = build_segmentor(cfg.model)
ckpt = torch.load(CKPT, map_location='cpu')
_ = load_checkpoint(model, CKPT, map_location='cpu', strict=False)
model_keys = set(model.state_dict().keys())
ckpt_keys = set(ckpt['state_dict'].keys())
missing = model_keys - ckpt_keys
unexpected = ckpt_keys - model_keys
print(f'[ckpt] model keys={len(model_keys)}, ckpt keys={len(ckpt_keys)}, '
      f'missing={len(missing)}, unexpected={len(unexpected)}')
if missing:
    print('[ckpt] MISSING sample:', sorted(missing)[:8])
if unexpected:
    print('[ckpt] UNEXPECTED sample:', sorted(unexpected)[:8])
model = MMDataParallel(model, device_ids=[0])

# ---------------------------------------------------------------------------
# 6. 数据集与评测
# ---------------------------------------------------------------------------
dataset = build_dataset(cfg.data.test)
SMOKE = int(os.environ.get('SMOKE', '0'))
if SMOKE:
    dataset.img_infos = dataset.img_infos[:SMOKE]
    dataset.len = SMOKE
    print(f'[smoke] only first {SMOKE} images')
loader = build_dataloader(dataset, samples_per_gpu=1,
                          workers_per_gpu=cfg.data.workers_per_gpu,
                          dist=False, shuffle=False)

results = single_gpu_test(model, loader)
metrics = dataset.evaluate(results, metric='mIoU')
round_tag = 'r2_sfm' if SFM_INJECT else 'r1_orig'
print(f'\n===== Cityscapes CCNet-R50-D16 评测结果（{round_tag}）=====')
for k, v in metrics.items():
    if isinstance(v, float):
        print(f'{k}: {v:.4f}')
out_file = os.path.join(WORKDIR, f'eval_cs_{round_tag}.txt')
with open(out_file, 'w') as f:
    for k, v in metrics.items():
        f.write(f'{k}\t{v}\n')
print(f'[done] saved to {out_file}')

# ---------------------------------------------------------------------------
# 7. SAVE_RESULTS=1: 落盘预测 + 诊断统计（疑案排查：预测 shape / 四类像素 / 逐张 vs GT）
# ---------------------------------------------------------------------------
if os.environ.get('SAVE_RESULTS', '0') == '1':
    import pickle
    from collections import Counter
    import numpy as np

    squeezed = [np.asarray(r).squeeze().astype(np.uint8) for r in results]
    shapes = Counter(a.shape for a in squeezed)
    print('[diag] 预测 shape 分布:', dict(shapes))
    cnt_pred = np.zeros(19, dtype=np.int64)
    for a in squeezed:
        cnt_pred += np.bincount(a.ravel(), minlength=19)[:19]
    print('[diag] 预测 19 类像素总数:', cnt_pred.tolist())
    cnt_gt = np.zeros(19, dtype=np.int64)
    for i in range(len(dataset)):
        g = dataset.get_gt_seg_map_by_idx(i)
        cnt_gt += np.bincount(g[g != 255].ravel(), minlength=19)[:19]
    print('[diag] GT     19 类像素总数:', cnt_gt.tolist())
    with open(os.path.join(WORKDIR, f'results_{round_tag}.pkl'), 'wb') as f:
        pickle.dump(squeezed, f, protocol=4)
    print(f'[diag] results saved to results_{round_tag}.pkl')

