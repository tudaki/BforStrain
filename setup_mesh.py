import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tools import make_patches_creep, llh2local, local2llh, lap_smooth, point_force
from scipy.spatial import Voronoi, Delaunay
import pandas as pd
import trimesh

savename = "mesh_WUS_Johnson2023"
origin = [34, -120] 

#format of columns:
#   lon       lat       Ve       Vn       Se       Sn  
data = np.loadtxt('Zeng_vels_augmented_PA_JDF.txt',delimiter=',')


# format of columns:
# fault id, lon endpoint 1, lat endpoint 1, lon endpoint 2, lat endpoint 2, creep rate (mm/yr)
# note fault id is an integer, a unique identifier for each fault that section is considered to be a "continuous fault" 
creeping_faults = np.loadtxt('creeping_faults.txt',delimiter=',')

patchL = 15.

lon_range = [-127, -96]
lat_range = [26, 54]
nom_node_spacing = 100.
#option to refine mesh in vicinity of GPS data
#specifiy level of refinement, integer from 0 to 4
#level 0 means no refinement, 4 is maximum refinement
refine_mesh = 0

#unlike matlab, when defined this way we DON'T have to transpose llh
llh=np.array([data[:,0],data[:,1],np.zeros(data[:,1].shape)])
xy_gps = llh2local(llh,[origin[1],origin[0]]).T
Ve, Vn, Sige, Sign = data[:, 2], data[:, 3], data[:, 4], data[:, 5]

bxy = llh2local(np.array([lon_range,lat_range]),[origin[1],origin[0]]).T
minx, maxx = np.min(bxy, axis=0)[0], np.max(bxy, axis=0)[0]
miny, maxy = np.min(bxy, axis=0)[1], np.max(bxy, axis=0)[1]

nx=round((maxx-minx)/nom_node_spacing)
ny=round((maxy-miny)/nom_node_spacing)
x=np.linspace(minx,maxx,nx)
y=np.linspace(miny,maxy,ny)
X,Y=np.meshgrid(x,y)
nodes = np.column_stack((X.ravel(), Y.ravel()))

ind_gps = (xy_gps[:, 0] < maxx) & (xy_gps[:, 0] > minx) & (xy_gps[:, 1] < maxy) & (xy_gps[:, 1] > miny)

xy_subset = xy_gps[ind_gps]
# Compute the Voronoi diagram
vor = Voronoi(xy_subset)
# Extract the vertices of the Voronoi diagram
vx = vor.vertices[:, 0]
vy = vor.vertices[:, 1]
_, unique_indices = np.unique(vx, return_index=True) 
vx = vx[unique_indices]
vy = vy[unique_indices]

ind_vor = (vx < maxx) & (vx > minx) & (vy < maxy) & (vy > miny)
nodes_refine = np.column_stack((vx[ind_vor], vy[ind_vor]))

if refine_mesh == 4:
    nodes = np.vstack((nodes, xy_gps[ind_gps], nodes_refine))
elif refine_mesh == 3:
    nodes = np.vstack((nodes, xy_gps[ind_gps], nodes_refine[::2]))
elif refine_mesh == 2:
    nodes = np.vstack((nodes, xy_gps[ind_gps], nodes_refine[::3]))
elif refine_mesh == 1:
    nodes = np.vstack((nodes, xy_gps[ind_gps], nodes_refine[::5]))
else:
    nodes = np.vstack((nodes, xy_gps[ind_gps]))

if creeping_faults.size != 0:
    edge_creep, node_creep, PatchEnds, PatchCreepRates, SegEnds = make_patches_creep(creeping_faults,origin,minx,maxx,miny,maxy,patchL)

else:
    node_creep=[]
    SegEnds = []
    PatchEnds = []
    PatchCreepRates = []

nodes = np.vstack((nodes, node_creep))

triDel=Delaunay(nodes, qhull_options = 'Qt Qbb Qc') #qhull settings consistent with matlab
tri=triDel.simplices

smoothed_nodes, _ = lap_smooth(nodes,tri,5) #third argument is number of iterations
nodes = smoothed_nodes
elts=nodes[tri]

centroids = np.mean(elts, axis=1)
vec1 = elts[:, 1, :] - elts[:, 0, :]
vec2 = elts[:, 2, :] - elts[:, 0, :]
areas = 0.5 * np.abs(np.cross(vec1, vec2))

# Toss out GPS data outside of mesh domain
xy_gps, Ve, Vn, Sige, Sign = (arr[ind_gps] for arr in (xy_gps, Ve, Vn, Sige, Sign))
%matplotlib inline
import mpld3
mpld3.enable_notebook()
#plt.tripcolor(nodes[:,0], nodes[:,1],triDel.simplices.copy(),linewidth=0.25,facecolors=areas)
plt.triplot(nodes[:,0], nodes[:,1],triDel.simplices.copy(),linewidth=0.25)
plt.quiver(xy_gps[:,0],xy_gps[:,1],Ve,Vn, color='red')
plt.plot(PatchEnds[:, [0, 2]].T, PatchEnds[:, [1, 3]].T, color='blue')  # Plot all line segments at once
plt.gca().set_aspect('equal')
plt.show()

