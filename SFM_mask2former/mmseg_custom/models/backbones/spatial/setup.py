'''
Modified from scipy.spatial
'''




from __future__ import division, print_function, absolute_import

from os.path import join, dirname
import glob
import subprocess
import os
import sys

def pre_build_hook(build_ext, ext):
    from scipy._build_utils.compiler_helper import get_cxx_std_flag
    std_flag = get_cxx_std_flag(build_ext._cxx_compiler)
    if std_flag is not None:
        ext.extra_compile_args.append(std_flag)

def gen_cython():
    cwd = os.path.abspath(os.path.dirname(__file__))
    print("Cythonizing sources")
    p = subprocess.call([sys.executable,
                         os.path.join(cwd, 'cythonize.py'),
                         '.'],
                        cwd=cwd)
    if p!=0:
        raise RuntimeError('Cythonizing Failed!')

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    from numpy.distutils.misc_util import get_info as get_misc_info
    from scipy._build_utils.system_info import get_info as get_sys_info
    from distutils.sysconfig import get_python_inc
    # from scipy.spatial import transform

    config = Configuration('spatial', parent_package, top_path)

    config.add_data_dir('tests')

    # spatial.transform
    config.add_subpackage('transform')

    # qhull
    qhull_src = sorted(glob.glob(join(dirname(__file__), 'qhull_src',
                                    'src', '*.c')))

    inc_dirs = [get_python_inc()]
    if inc_dirs[0] != get_python_inc(plat_specific=1):
        inc_dirs.append(get_python_inc(plat_specific=1))
    inc_dirs.append(get_numpy_include_dirs())
    inc_dirs.append(join(dirname(dirname(__file__)), '_lib'))

    cfg = dict(get_sys_info('lapack_opt'))
    cfg.setdefault('include_dirs', []).extend(inc_dirs)
    config.add_extension('qhull',
                         sources=['qhull.c', 'qhull_misc.c'] + qhull_src,
                         **cfg)

    # _distance_wrap
    config.add_extension('_distance_wrap',
                         sources=[join('src', 'distance_wrap.c')],
                         depends=[join('src', 'distance_impl.h')],
                         include_dirs=[get_numpy_include_dirs()],
                         extra_info=get_misc_info("npymath"))

    config.add_extension('_voronoi',
                         sources=['_voronoi.c'])

    config.add_extension('_hausdorff',
                         sources=['_hausdorff.c'])

    # Add license files
    config.add_data_files('qhull_src/COPYING.txt')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    gen_cython()
    setup(**configuration(top_path='').todict())
