
def test_make_patches_creep():
    import numpy as np

    origin = [34, -120] 
    creeping_faults = np.loadtxt('creeping_faults.txt',delimiter=',')

    minx=-700.4920548113531 
    maxx=1543.891117953848 
    miny=-868.0390077866507 
    maxy=2486.3562032891
    patchL=15.

    edge_creep, node_creep, PatchEnds, PatchCreepRates, SegEnds = make_patches_creep(creeping_faults,origin,minx,maxx,miny,maxy,patchL)

    print(sum(PatchEnds))


def lap_smooth(nodes,tri,iterations):
    import numpy as np
    # Perform Laplacian smoothing
    for _ in range(iterations):
        smoothed_nodes = np.copy(nodes)
        for i in range(len(nodes)):
            neighbor_indices = np.where(np.any(tri == i, axis=1))
            neighbors = nodes[np.unique(tri[neighbor_indices])]
            avg_x = np.mean(neighbors[:, 0])
            avg_y = np.mean(neighbors[:, 1])
            smoothed_nodes[i] = [avg_x, avg_y]
        print("Smoothing iteration: ", _+1," of: ", iterations)
    return(smoothed_nodes,tri)

def make_triangular_patch_stuff(tri, p):
    strikevec_faces = []
    strike_faces = []
    dipvec_faces = []
    dip_faces = []
    centroids_faces = []
    normal_faces = []
    area_faces = []

    for j in range(tri.shape[0]):
        temp1 = [p]
        temp2 = [tri[j, :]]

        vec1 = p[temp2[0][0], :] - p[temp2[0][1], :]
        vec2 = p[temp2[0][2], :] - p[temp2[0][1], :]
        cross_face = np.cross(vec1, vec2)
        veclength = np.linalg.norm(cross_face)
        normal = cross_face / veclength
        strikevec = np.array([1, -normal[0] / normal[1], 0])
        strikevec = strikevec / np.linalg.norm(strikevec)
        dipvec = np.cross(normal, strikevec)

        if dipvec[2] > 0:
            dipvec = -dipvec
        if normal[2] < 0:
            normal = -normal
        strikevec = np.cross(normal, dipvec)

        normal_faces.append(normal)
        strikevec_faces.append(strikevec)
        dipvec_faces.append(dipvec)
        strike_faces.append(90 - np.arctan2(strikevec[1], strikevec[0]) * 180 / np.pi)
        dip_faces.append(np.abs(np.arctan(dipvec[2] / np.sqrt(dipvec[0] ** 2 + dipvec[1] ** 2)) * 180 / np.pi))
        centroids_faces.append([np.mean(temp1[0][temp2[0], 0]), np.mean(temp1[0][temp2[0], 1]), np.mean(temp1[0][temp2[0], 2])])
        area_faces.append(0.5 * np.abs(np.linalg.norm(np.cross(vec1, vec2))))

    patch_stuff = {
        'strikevec_faces': strikevec_faces,
        'strike_faces': strike_faces,
        'dipvec_faces': dipvec_faces,
        'dip_faces': dip_faces,
        'centroids_faces': centroids_faces,
        'normal_faces': normal_faces,
        'area_faces': area_faces
    }
    return patch_stuff


def make_patches_creep(creeping_faults,origin,minx,maxx,miny,maxy,patchL):
    import numpy as np
    import pandas as pd

    #Convert lat,long to x,y
    x = creeping_faults[:, 1:5]
    llhx = x[:, 0:2]
    x1 = llh2local(llhx.T, origin[::-1]).T
    llhx = x[:, 2:4]
    x2 = llh2local(llhx.T, origin[::-1]).T
    SegEnds = np.column_stack((x1,x2))
    
    # Remove segments outside the specified range
    rempatch = (SegEnds[:, 0] < minx) | (SegEnds[:, 0] > maxx) | (SegEnds[:, 1] < miny) | (SegEnds[:, 1] > maxy)
    rempatch = rempatch | (SegEnds[:, 2] < minx) | (SegEnds[:, 2] > maxx) | (SegEnds[:, 3] < miny) | (SegEnds[:, 3] > maxy)
    SegEnds = SegEnds[~rempatch]
    creeping_faults = creeping_faults[~rempatch]
    
    # Initialize variables
    PatchEnds = np.empty((0, 4), dtype=float)
    #PatchEnds = []
    PatchCreepRates = np.empty((0,1), dtype=float)
    #PatchCreepRates = []
    Patch_id = np.empty((0,1), dtype=int)

    for k in range(SegEnds.shape[0]):
        patchlength = np.sqrt((SegEnds[k, 2] - SegEnds[k, 0]) ** 2 + (SegEnds[k, 3] - SegEnds[k, 1]) ** 2)
        numpatch = int(np.ceil(patchlength / patchL))
        xs = np.linspace(SegEnds[k, 0], SegEnds[k, 2], numpatch + 1)
        ys = np.linspace(SegEnds[k, 1], SegEnds[k, 3], numpatch + 1)
        PatchEnds = np.vstack((PatchEnds, np.column_stack((xs[:-1], ys[:-1], xs[1:], ys[1:]))))
        PatchCreepRates = np.vstack((PatchCreepRates, np.full((numpatch, 1), creeping_faults[k, 5])))
        Patch_id = np.vstack((Patch_id, np.full((numpatch,1), creeping_faults[k, 5])))

    segends1 = PatchEnds[:, :2]
    segends2 = PatchEnds[:, 2:]

    num_rows = segends1.shape[0] + segends2.shape[0]
    node_creep = np.zeros((num_rows, 2))

    node_creep[::2] = segends1
    node_creep[1::2] = segends2

    # Create 'edge_creep' using NumPy
    edge_creep = np.column_stack((np.arange(0, num_rows, 2), np.arange(1, num_rows, 2)))

    # Convert the NumPy array to a pandas DataFrame because the numpy unique function automatically sorts
    df = pd.DataFrame(np.round(node_creep, 4), columns=['x', 'y'])
    ic, unique_vals = pd.factorize(df.apply(tuple, axis=1))
    node_creep = np.array(unique_vals.tolist())

    new_edge = edge_creep.copy()

    for k in range(len(ic)):
        new_edge[edge_creep == (k)] = ic[k]

    edge_creep = new_edge
    
    return edge_creep, node_creep, PatchEnds, PatchCreepRates, SegEnds

def llh2local(llh, origin):
    """
    Converts from longitude and latitude to local coordinates given an origin.

    Parameters:
        llh: Numpy array of shape (3,n) - lon (decimal degrees), lat (decimal degrees), height (ignored).
            Note that this will not work for an array of shape (3,) e.g. a one dimensional Numpy array.
            This could be added as a further test
        origin: Numpy array of shape (2,) - lon (decimal degrees), lat (decimal degrees).

    Returns:
        xy: Numpy array of shape (2,) - local coordinates in kilometers.

    Record of revisions:

    Date          Programmer            Description of Change
    ====          ==========            =====================

    Sept 7, 2000  Peter Cervelli		Original Code
    Oct 20, 2000  Jessica Murray        Changed name from DM_llh2local to 
                                        llh2local for use with non-DM functions;
                                        Added to help message to clarify order
                                        of 'llh' (i.e., lon, lat, height).
    Dec. 6, 2000  Jessica Murray        Clarified help to show that llh 
    Oct. 23, 2023 Jacob Dorsett         Converted code to Python3

    """
    import numpy as np

    # Set ellipsoid constants (WGS84)
    a = 6378137.0
    e = 0.08209443794970

    # Convert to radians
    llh = np.radians(llh)
    origin = np.radians(origin)
    # Do the projection
    z = llh[1] != 0

    dlambda = llh[0, z] - origin[0]

    M = a * (
        (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * llh[1, z]
        - (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * np.sin(2 * llh[1, z])
        + (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * np.sin(4 * llh[1, z])
        - (35 * e ** 6 / 3072) * np.sin(6 * llh[1, z])
    )
    M0 = a * (
        (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * origin[1]
        - (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * np.sin(2 * origin[1])
        + (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * np.sin(4 * origin[1])
        - (35 * e ** 6 / 3072) * np.sin(6 * origin[1])
    )

    N = a / np.sqrt(1 - e ** 2 * np.sin(llh[1, z]) ** 2)
    E = dlambda * np.sin(llh[1, z])
    xy = np.zeros((2, llh.shape[1]))

    xy[0, z] = N / np.tan(llh[1, z]) * np.sin(E)
    xy[1, z] = M - M0 + N / np.tan(llh[1, z]) * (1 - np.cos(E))

    # Handle special case of latitude = 0
    xy[0, ~z] = a * dlambda[~z]
    xy[1, ~z] = -M0

    # Convert to km
    xy = xy / 1000

    return xy




def local2llh(xy, origin):
    """
    Converts from local coordinates to longitude and latitude given the [lon, lat] of an origin.

    Parameters:
        xy: Numpy array or list of shape (2,) - local coordinates in kilometers.
        origin: Numpy array or list of shape (2,) - lon (decimal degrees), lat (decimal degrees).

    Returns:
        llh: Numpy array of shape (3,) - [lon, lat, height] in decimal degrees.

    Ported to Python by Jacob Dorsett. Currently returns incorrect values when I input multiple data points. But Matlab does too
    """

    import numpy as np

    # Set ellipsoid constants (WGS84)
    a = 6378137.0
    e = 0.08209443794970

    # Convert to radians / meters
    xy = np.array(xy) * 1000
    origin = np.radians(origin)

    # Iterate to perform inverse projection

    M0 = a * (
        (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * origin[1]
        - (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * np.sin(2 * origin[1])
        + (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * np.sin(4 * origin[1])
        - (35 * e ** 6 / 3072) * np.sin(6 * origin[1])
    )

    z = xy[1] != -M0

    A = (M0 + xy[1, z]) / a
    B = xy[0, z] ** 2 / a ** 2 + A ** 2

    llh = np.zeros((2, len(xy[0])))
    llh[1, z] = A
    delta = np.inf
    c = 0

    while np.max(np.abs(delta)) > 1e-8:
        C = np.sqrt((1 - e ** 2 * np.sin(llh[1, z]) ** 2)) * np.tan(llh[1, z])
        M = a * (
            (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * llh[1, z]
            - (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * np.sin(2 * llh[1, z])
            + (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * np.sin(4 * llh[1, z])
            - (35 * e ** 6 / 3072) * np.sin(6 * llh[1, z])
        )

        Mn = 1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256 - \
            2 * (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * np.cos(2 * llh[1, z]) + \
            4 * (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * np.cos(4 * llh[1, z]) - \
            6 * (35 * e ** 6 / 3072) * np.cos(6 * llh[1, z])

        Ma = M / a

        delta = -(A * (C * Ma + 1) - Ma - 0.5 * (Ma ** 2 + B) * C) / \
            (e ** 2 * np.sin(2 * llh[1, z]) * (Ma ** 2 + B - 2 * A * Ma) / (4 * C) + (A - Ma) * (C * Mn - 2 / np.sin(2 * llh[1, z])) - Mn)

        llh[1, z] = llh[1, z] + delta
        c = c + 1
        if c > 100:
            raise ValueError('Convergence failure.')

    llh[0, z] = (np.arcsin(xy[0, z] * C / a) / np.sin(llh[1, z])) + origin[0]

    # Handle special case of latitude = 0
    llh[0, ~z] = xy[0, ~z] / a + origin[0]
    llh[1, ~z] = 0

    # Convert back to decimal degrees
    llh = np.degrees(llh)

    return llh



def test_llh2local():
    import numpy as np
    llh_in = np.array([[-117.09, 34.116,0],
                    [-118.14, 33.804,0],
                    [-117.5, 34.2,0],
                    [-116.8, 33.4,0],])
    origin = np.array([34, -120])

    xys_out=llh2local(llh_in.T,origin)
    #print(xys_out.T)

def test_local2llh():
    import numpy as np
    
    origin = np.array([34, -120])
    #latlons=local2llh(xys/1000,origin)
    xys_in = np.array([[-9385.14062638,25687.27286049],
                         [-9494.4196537, 25737.44917358],
                         [-9361.58957964,25737.23996962],
                         [-9610.92767148,25551.67046323]])
    


    llh_out = local2llh(xys_in/1000,origin)

    print(llh_out)


def point_force(xs, ys, x, y, nu):
    
    import numpy as np

    # Shift coordinates
    x = x - xs
    y = y - ys
    r = np.sqrt(x ** 2 + y ** 2)

    q = (3 - nu) * np.log(r) + (1 + nu) * y ** 2 / r ** 2
    p = (3 - nu) * np.log(r) + (1 + nu) * x ** 2 / r ** 2
    w = -(1 + nu) * x * y / r ** 2

    Ue_x = q
    Ue_y = w
    Un_x = w
    Un_y = p

    # Strain rates
    dq_dx = (3 - nu) * x / r ** 2 - 2 * (1 + nu) * y ** 2 * x / r ** 4
    dq_dy = (3 - nu) * y / r ** 2 + 2 * (1 + nu) * (-y ** 3 / r ** 4 + y / r ** 2)

    dp_dx = (3 - nu) * x / r ** 2 + 2 * (1 + nu) * (-x ** 3 / r ** 4 + x / r ** 2)
    dp_dy = (3 - nu) * y / r ** 2 - 2 * (1 + nu) * x ** 2 * y / r ** 4

    dw_dx = -(1 + nu) * (-2 * x ** 2 * y / r ** 4 + y / r ** 2)
    dw_dy = -(1 + nu) * (-2 * y ** 2 * x / r ** 4 + x / r ** 2)

    Exx_x = dq_dx
    Eyy_x = dw_dy
    Exy_x = 0.5 * (dw_dx + dq_dy)

    Exx_y = dw_dx
    Eyy_y = dp_dy
    Exy_y = 0.5 * (dp_dx + dw_dy)

    # Rotation rate
    omega_x = 0.5 * (dq_dy - dw_dx)
    omega_y = 0.5 * (dw_dy - dp_dx)

    return Ue_x, Ue_y, Un_x, Un_y, Exx_x, Exy_x, Eyy_x, Exx_y, Exy_y, Eyy_y, omega_x, omega_y


if __name__ == '__main__':
    #test_llh2local() #this one is equivalent to matlab! done
    #test_local2llh() this one also is equivalent to matlab... but both are wrong?
    test_make_patches_creep()