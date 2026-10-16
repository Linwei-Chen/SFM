# -*- coding: utf-8 -*-
"""最小化的 scipy.spatial 子集：只保留 qhull（Delaunay 三角剖分）。

源码取自 scipy ``spatial`` 模块（Pauli Virtanen 等，BSD 许可），原始项目
仅使用了其中的 ``qhull.Delaunay``（MSAU 非均匀上采样的三角剖分引擎），
因此这里只编译/携带 qhull 一个扩展，其余子模块（ckdtree/voronoi/…）不带。
"""
from . import qhull
from .qhull import (Delaunay, ConvexHull, Voronoi, HalfspaceIntersection,
                    tsearch)

__all__ = ['qhull', 'Delaunay', 'ConvexHull', 'Voronoi',
           'HalfspaceIntersection', 'tsearch']
