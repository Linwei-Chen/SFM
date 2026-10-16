# -*- coding: utf-8 -*-
"""sfm.vendor —— 随仓库携带的第三方源码（仅在安装时可选编译，不影响纯 Python 使用）。

当前只包含 ``spatial``（scipy 同源的 qhull Cython 封装，BSD 许可），
用于 MSAU 的 qhull 非均匀上采样，见 :mod:`sfm.interpolate`。
"""
