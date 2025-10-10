#from __future__ import absolute_impor
import qhull
import numpy as np

img = np.random.random([100,100])
points = np.random.random([20,2])*100

h=100
w=100
row_coord = np.arange(h).repeat([w]).reshape([h,w])
col_coord = np.arange(w).repeat([h]).reshape([w,h]).T
coord = np.stack([row_coord, col_coord])
coord = coord.transpose([1,2,0]).reshape([-1,2])
tri = qhull.Delaunay(points)
tri_map, c = tri.find_simplex(coord, return_c = True)
print(tri_map)
print(c)
