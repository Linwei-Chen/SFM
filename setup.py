#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""sfm 安装脚本：pip install . 即完成安装，安装后直接 ``import sfm`` 调库使用。

包主体是纯 Python；唯一可选的 C 扩展是 ``sfm.vendor.spatial.qhull``
（scipy 同源 qhull Cython 源码，FCN style MSAU 非均匀上采样的 bit-exact 后端）。
编译需要 C 编译器 + LAPACK：安装时自动探测，探测/编译失败都会**优雅跳过**，
运行时自动回退到 ``scipy.spatial.Delaunay`` 实现（同一 Qhull 引擎），
不影响 ``import sfm`` 与其余功能。
"""
from setuptools import setup, find_packages

import os
import sys


def _bundled_openblas():
    """找 numpy/scipy 等 pip 轮子里捆绑的 OpenBLAS（自带 LAPACK 符号）。

    返回 ``(lib_dir, lib_basename)`` 或 ``None``。轮子把它放在
    ``site-packages/<pkg>.libs/libopenblas*.so*``，链接用 ``-l:文件名`` 加
    rpath 指向该目录即可，无需系统安装任何 BLAS。
    """
    import glob as _glob
    from sysconfig import get_paths

    cand_dirs = [get_paths()['purelib']]
    for pkg_dir in list(cand_dirs):
        cand_dirs += _glob.glob(os.path.join(pkg_dir, '*.libs'))
    for d in cand_dirs:
        for path in sorted(_glob.glob(os.path.join(d, 'libopenblas*.so*'))) + \
                sorted(_glob.glob(os.path.join(d, 'libopenblas*.dylib'))):
            return d, os.path.basename(path)
    return None


def _lapack_link_info():
    """探测本机可链接的 LAPACK（qhull 变换矩阵计算需要 dgetrf/dgecon/dgetrs）。

    返回 ``(libraries, extra_link_args)``，探测不到返回 ``(None, None)``。
    """
    # 1) numpy/scipy 轮子捆绑的 OpenBLAS（无任何系统依赖时也能编）
    bundled = _bundled_openblas()
    if bundled is not None:
        lib_dir, lib_name = bundled
        return [], ['-L' + lib_dir, '-l:' + lib_name, '-Wl,-rpath,' + lib_dir]
    # 2) numpy.distutils（numpy<1.23 的训练环境走这里，最贴近原版 setup）
    try:
        from numpy.distutils.system_info import get_info
        info = get_info('lapack_opt')
        if info.get('libraries'):
            return info['libraries'], []
    except Exception:
        pass
    # 3) 真实链接探测（系统 lapack/blas）
    from setuptools._distutils.ccompiler import new_compiler
    from setuptools._distutils.errors import CompileError, LinkError
    from setuptools._distutils.sysconfig import customize_compiler

    cc = new_compiler()
    customize_compiler(cc)
    candidates = [['lapack'], ['lapack', 'blas']]
    if sys.platform == 'darwin':
        candidates.insert(0, [])
    for libs in candidates:
        link_args = (['-framework', 'Accelerate']
                     if sys.platform == 'darwin' and not libs else [])
        try:
            if cc.has_function('dgetrf_', libraries=libs,
                               extra_link_args=link_args):
                return libs, link_args
        except (CompileError, LinkError):
            continue
        except Exception:
            continue
    return None, None


def _qhull_extension():
    """构造 vendored qhull 扩展；环境不具备条件时返回 None（纯 Python 安装）。"""
    root = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(root, 'sfm', 'vendor', 'spatial')
    if not os.path.exists(os.path.join(src_dir, 'qhull.c')):
        return None
    import glob
    sources = ['sfm/vendor/spatial/qhull.c', 'sfm/vendor/spatial/qhull_misc.c']
    sources += sorted(glob.glob('sfm/vendor/spatial/qhull_src/src/*.c'))
    include_dirs = [src_dir, os.path.join(src_dir, 'qhull_src', 'src')]
    try:
        import numpy
        include_dirs.append(numpy.get_include())
    except ImportError:
        return None
    libs, extra_link_args = _lapack_link_info()
    if libs is None:
        print('[sfm] 未探测到可链接的 LAPACK，跳过 vendored qhull 扩展编译；'
              '运行时将使用 scipy.spatial.Delaunay 回退实现')
        return None
    from setuptools import Extension
    return Extension('sfm.vendor.spatial.qhull', sources=sources,
                     include_dirs=include_dirs, libraries=libs,
                     extra_link_args=extra_link_args)


def make_build_ext_cls():
    """单个扩展编译失败不影响整个安装（qhull 扩展是可选加速）。"""
    try:
        from setuptools.command.build_ext import build_ext
    except ImportError:
        return None

    class optional_build_ext(build_ext):
        def build_extensions(self):
            for ext in self.extensions:
                try:
                    self.build_extension(ext)
                except Exception as e:
                    print('[sfm] WARNING: vendored qhull 扩展编译失败（%s）；'
                          '运行时将使用 scipy.spatial 回退实现' % e)

    return optional_build_ext


def main():
    packages = find_packages(include=('sfm', 'sfm.*'))
    ext_modules = []
    cmdclass = {}
    ext = _qhull_extension()
    if ext is not None:
        ext_modules = [ext]
        cls = make_build_ext_cls()
        if cls is not None:
            cmdclass = {'build_ext': cls}

    setup(
        name='sfm',
        version='1.1.0',
        description=('SFM: Spatial Frequency Modulation plug-and-play operators '
                     '(ARS adaptive sampling; qhull non-uniform upsampling for '
                     'FCN-style MSAU; SFM-FPN aligned upsampling) for '
                     'MMDetection / MMSegmentation'),
        long_description=(
            'Plug-and-play core operators of the TPAMI 2025 paper '
            '"Spatial Frequency Modulation for Semantic Segmentation". '
            'Two release lines: (1) FCN-style qhull-MSAU (ARS + non-uniform '
            'upsampling via Delaunay triangulation + LPRM); (2) FPN line '
            '(ARS + CARAFE-style reassembly + 3-point dynamic sampling '
            'weighted alignment, e.g. Mask2Former). '
            'Import `sfm` to auto-register `SFMFPN` into mmdetection/'
            'mmsegmentation neck registries; pure PyTorch usage also supported.'),
        long_description_content_type='text/plain',
        author='Linwei Chen',
        author_email='chenlinwei@csu.edu.cn',
        url='https://github.com/Linwei-Chen/SFM',
        license='Apache License 2.0',
        packages=packages,
        exclude_package_data={'': ['tests', 'examples']},
        python_requires='>=3.7',
        # 纯 Python 包；依赖 torch（训练环境均已自带），不在此硬性声明以免
        # pip 在无 torch 的环境里触发重安装。scipy 仅在 qhull 扩展未编译时
        # 作为回退后端需要，运行时才检查。
        install_requires=[],
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        keywords=['semantic-segmentation', 'object-detection', 'fpn', 'carafe',
                  'adaptive-sampling', 'frequency', 'qhull', 'msau',
                  'mmdetection', 'mmsegmentation'],
        classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
    )


if __name__ == '__main__':
    main()
