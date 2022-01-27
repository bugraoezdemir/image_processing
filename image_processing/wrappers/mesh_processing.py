

"""The aim of this module is to combine power of several packages to convert 3D numpy arrays into
    surface meshes, apply several analyses on them and visualise them. The module takes advantage of
    the following packages: numpy, scipy, pyvista, pymeshfix, igl and gdist. The module is particularly
    focused on 3D B-cell images from lattice light sheet microscope."""


import numpy as np
import vtk
import pyvista as pv
import pandas as pd
import igl
import scipy as sp
from vtk.util.numpy_support import vtk_to_numpy
from scipy.sparse.linalg import spsolve
from scipy import ndimage as ndi
import pymeshfix as meshfix
# import gdist
# from vedo.mesh import Mesh as vtkmesh
# from vedo import embedWindow
# import vedo
from matplotlib import cm
import os


### local imports
from image_processing.utils import convenience as cnv


# embedWindow(backend=False)

def vtk_cellarray_to_shape(vtk_cellarray, ncells):
    """Turn a vtkCellArray into a numpyarray of a fixed shape
        assumes your cell array has uniformed sized cells

    Parameters
    ----------
    vtk_cellarray : vtk.vtkCellArray
        a cell array to convert
    ncells: int
        how many cells are in array

    Returns
    -------
    cellarray: array
        cellarray, a ncells x K array of cells, where K is the
        uniform shape of the cells.  Will error if cells are not uniform
    """
    cellarray = vtk_to_numpy(vtk_cellarray)
    cellarray = cellarray.reshape(ncells, int(len(cellarray)/ncells))
    return cellarray[:, 1:]


def poly_to_mesh_components(poly):
    """ converts a vtkPolyData to its numpy components
    Parameters
    ----------
    poly : vtk.vtkPolyData
        a polydate object to convert to numpy components
    Returns
    -------
    np.array
        points, the Nx3 set of vertex locations
    np.array
        tris, the KxD set of faces (assumes a uniform cellarray)
    np.array
        edges, if exists uses the GetLines to make edges
    """
    points = vtk_to_numpy(poly.GetPoints().GetData())
    ntris = poly.GetNumberOfPolys()
    if ntris > 0:
        tris = vtk_cellarray_to_shape(poly.GetPolys().GetData(), ntris)
    else:
        tris = None
    nedges = poly.GetNumberOfLines()
    if nedges > 0:
        edges = vtk_cellarray_to_shape(poly.GetLines().GetData(), nedges)
    else:
        edges = None
    return (points, tris, edges)


def compute_cell_size(mesh):
    alg = vtk.vtkCellSizeFilter()
    alg.SetInputDataObject(mesh)
    alg.Update()
    nmsh = pv.wrap(alg.GetOutputDataObject(0))
    nmsh = nmsh.ctp()
    return nmsh



def _is_binary(array):
    cnv.cp_array(array)    
    return (len(np.unique(array)) == 2)


class griddata(pv.UniformGrid):
    def __init__(self):
        super(griddata, self).__init__()
        self.grid_exists = False
        """ Nothing needed here. """
    def array_to_grid(self, array, sampling = None, name = None, grid_spacing = (1, 1, 1)): ### SU NAMING OLAYINI NETLESTIR.
        """ Set a numpy array volume as the main input of the class. If the volume is binary,
            first a distance transform will be executed and the distance map will be used for
            contouring. If a greyscale volume is used, then the contouring will be directly 
            on this grey-valued array."""
        self.dimensions = array.shape
        self.origin = (0,0,0)
        self.spacing = grid_spacing
        if name is not None:
            self.object_name = name
        self.array = array.copy()
        if sampling is None:
            self.sampling = (1, 1, 1)
        else:
            self.sampling = sampling
        # print(np.unique(self.binary))
        if _is_binary(self.array):
            item = ndi.distance_transform_edt(self.array, sampling = self.sampling)
            if name is None:
                name = 'depth_map'
            self.point_data[name] = item.flatten(order="F")
            print('Binary volume assigned. Distance transform executed. Grid created.')
        else:
            item = self.array.copy()
            if name is None:
                name = 'scalar_map'
            self.point_data[name] = item.flatten(order="F")
            print('Greyscale volume assigned. Grid created.')
        self.grid_exists = True
    def add_scalars(self, scalar_name, array):
        cnv.cp_array(array)
        assert tuple(self.dimensions) == tuple(array.shape), 'input array must have the same shape as grid.'
        self.point_data[scalar_name] = array.flatten(order = 'F')        
    def to_surf(self, isosurface = 1, scalar_name = 'depth_map'):
        """This is pyvista contour finding algorithm to find triangular surface."""
        assert self.grid_exists, 'A reference volume is needed to contour. This is mostly a binary volume.'
        if _is_binary(self.array):
            if scalar_name is None:
                scalar_name = 'depth_map'
        tria = self.contour(isosurfaces = [isosurface], scalars = scalar_name, compute_normals = False, 
                             compute_gradients = False, compute_scalars = True, rng = None, 
                             preference = 'point', method = 'contour', progress_bar = False)
        print('Surface extraction complete. Might need reparation.')
        return tria
        


class surfdata(pv.PolyData):
    def __init__(self):
        super(surfdata, self).__init__()
        self.griddata_exists = False
        self.polydata_exists = False
        self.fixed = False
    def from_grid(self, grid, scalar_name = 'depth_map', isosurface = 1):
        """ This is the core method that interplays between grid and triangle meshes. """ 
        assert hasattr(grid, 'to_surf'), 'Input must be a griddata object'
        self.grid = grid
        self.griddata_exists = True        
        tria = self.grid.to_surf(isosurface, scalar_name)
        self.merge_with(tria)
    def from_array(self, array, grid_spacing, scalar_name, isosurface):
        """ This method has two goals:
                1. Create a grid from numpy array. This grid holds the properties of the 3D object, such as shape and spacing. 
                2. Contour the grid to create the triangle mesh.                
                """
        grid = griddata()
        grid.array_to_grid(array, sampling = (1, 1, 1), name = scalar_name, grid_spacing = grid_spacing)
        self.from_grid(grid, scalar_name, isosurface)
    def transfer_scalars(self, tria):
        """ Simply incorporate all point_data arrays from another triangular 
            mesh using the original scalar name. """
        for item in tria.point_data:
            try:
                self.point_data[item] = tria.point_data[item]  
            except:
                print('The point data length for ' + item + ' is inconsistent: ' + str(len(tria.point_data[item])))
                print('deleting: ' + item)
                self.point_data.pop(item)
    def merge_with(self, tria):
        """ Merge the existing polydata with another triangular mesh. If no triangular mesh exists yet,
            This method creates a polydata by copying all parameters from the input mesh. """
        self.points = tria.points
        self.faces = tria.faces
        self.transfer_scalars(tria)
        self.polydata_exists = True ### Not finished this method yet.
        if self.griddata_exists:
            self.sample_data()
    def sample_data(self, scalar_name = 'curvatures', mesh = None): 
        """ Sample from either of the three types of data: np.array, UniformGrid or PolyData. """
        if hasattr(mesh, 'nonzero'):
            ### Means an array is being sampled.
            ### If an array is being sampled, there has to be an existing grid already.
            assert self.griddata_exists, 'Cannot sample a volume without existing griddata.' 
            ### Make sure the existing grid and the new input array have the same shape 
            assert tuple(self.grid.dimensions) == tuple(mesh.shape)
            self.grid.add_scalars(scalar_name, mesh)
            ## After adding array to grid, simply sample the grid.
            sampled = self.sample(self.grid, tolerance = None, pass_cell_arrays = True, pass_point_data = True)            
            self.transfer_scalars(sampled)
        else: ### If the input is not an array
            if mesh is None: ### If no explicit new input, sample the existing grid. If no grid, produce an error and terminate.
                assert self.griddata_exists, 'Cannot sample a volume without existing griddata.'
                sampled = self.sample(self.grid, tolerance = None, pass_cell_arrays = True, pass_point_data = True)            
                self.transfer_scalars(sampled)
            elif hasattr(mesh, 'dimensions'):  ### If input is a new grid, transfer its scalars to existing grid.
                ### Make sure there is already a grid.
                assert self.griddata_exists, 'Cannot sample a new grid without existing griddata.'
                ### Make sure the new input grid and the existing grid have the same shape.
                assert tuple(self.grid.dimensions) == tuple(mesh.dimensions), 'New grid must have the same shape as existing grid.'
                for item in mesh.point_data: ### Transfer the scalars to grid
                    self.grid.add_scalars(item, mesh.point_data[item])
                ### After adding scalars, sample the grid.
                sampled = self.sample(self.grid, tolerance = None, pass_cell_arrays = True, pass_point_data = True)            
                self.transfer_scalars(sampled)
            elif hasattr(mesh, 'curvature'): ### If the input is a triangular mesh, simply transfer its scalars.
                self.transfer_scalars(mesh)
    def add_scalars(self, scalar_name, scalars):
        try:
            self.point_data[scalar_name] = scalars
        except:
            print('The scalar length is not fitting for the point_data: ' + str(len(scalars)))
    def centroid(self):
        return self.points.mean(axis = 0)
    def frameshift(self, vector):
        self.points = self.points - vector
        return self
    def to_origin(self):
        self.frameshift(self.centroid())
        return self
    def sas(self, scalar_name):
        """ Abbreviation for 'set_active_scalars' """
        self.set_active_scalars(scalar_name, preference = 'point')
        print('active scalars: ' + self.active_scalars_name)
        return self
    def get_trimesh(self, scalar_name = None):
        assert self.polydata_exists, 'No polydata. Either import or construct one.'
        if scalar_name is None:
            scalar_name = 'depth_map'
        verts, faces, _ = poly_to_mesh_components(self)
        scalars = self.point_data[scalar_name]
        return verts, faces, scalars 
    def largest(self):
        scalar_name = self.active_scalars_name
        self.extract_largest(inplace = True)
        self.sas(scalar_name)
        return self
    def fix(self):
        fix = meshfix.MeshFix(self)
        fix.repair()
        tria = fix.mesh.clean()
        self.merge_with(tria)
        self.fixed = True
        return self
    def clip_with_scalar(self, value, invert = False):
        clipped = self.clip_scalar(value = value, invert = invert)
        self.merge_with(clipped)
        return self
    def remove_small_components(self, size):        
        con = self.connectivity()
        pts = con.points
        scalars = con.point_data['RegionId']
        stack = np.hstack((scalars.reshape(-1, 1), pts))
        df = pd.DataFrame(stack, columns = ['region', 'depth', 'row', 'col'])
        grouped = df.groupby('region').transform('count')
        mask = grouped.depth < size
        inds = mask.index[mask].to_numpy()
        final = con.remove_points(inds)[0]
        self.merge_with(final)
    def rsc(self, size):        
        self.remove_small_components(size)
        return self        
    
    
    
        
    
class topograph(surfdata):
    def __init__(self):
        super(topograph, self).__init__()
    def calc_topography(self, rad = 5): 
        if not self.fixed:
            self.fix()
        if hasattr(self, 'shi'):
            print('Previously calculated topography detected. If the same radius, overwriting existing topography.')
        self.rad = rad
        verts, faces, _ = self.get_trimesh()
        v1, v2, pc1, pc2 = igl.principal_curvature(verts, faces, self.rad) # Principal curvatures
        self.add_scalars('minimum_rad_{}'.format(rad), pc1)
        self.add_scalars('maximum_rad_{}'.format(rad), pc2)
        gc = pc1 * pc2         # Gaussian curvature
        self.add_scalars('gaussian_rad_{}'.format(rad), gc)
        mc = 0.5 * (pc1 + pc2) # Mean curvature   
        self.add_scalars('mean_rad_{}'.format(rad), gc)
        e = np.e
        m = igl.massmatrix(verts, faces, igl.MASSMATRIX_TYPE_VORONOI)
        minv = sp.sparse.diags(1 / m.diagonal())
        ngc = minv.dot(gc)                        # Normalised gaussian curvature
        nmc = minv.dot(mc)                        # Normalised mean curvature
        self.add_scalars('normalised_gaussian_rad_{}'.format(rad), ngc)
        self.add_scalars('normalised_mean_rad_{}'.format(rad), nmc)        
        l = igl.cotmatrix(verts, faces)
        hn = -minv.dot(l.dot(verts))
        mcn = np.nan_to_num(np.linalg.norm(hn, axis=1))                      # Mean curvature normal
        blobness = np.nan_to_num((pc2 ** 2) / pc1)
        curvedness = np.nan_to_num(0.5 * np.sqrt((pc1 ** 2 + pc2 ** 2)))        
        self.add_scalars('mean_curvature_normal_rad_{}'.format(rad), mcn)
        self.add_scalars('blobness_rad_{}'.format(rad), blobness)
        self.add_scalars('curvedness_rad_{}'.format(rad), curvedness)
        K = (pc1 + pc2) / (pc1 - pc2)
        shi = -np.nan_to_num((2 / np.pi) * (np.arctan(K)))
        mshi = np.nan_to_num((2 * K) / (1 + K ** 2))
        doi1 = (pc2 - pc1) * mshi
        doi2 = K * (e ** (1 - np.abs(K)))
        doi3 = - (pc1 + pc2) * (e ** (1 - np.abs(K)))
        self.add_scalars('shape_index_rad_{}'.format(rad), shi)
        self.add_scalars('modified_shape_index_rad_{}'.format(rad), mshi)
        self.add_scalars('degree_of_interest1_rad_{}'.format(rad), doi1)
        self.add_scalars('degree_of_interest2_rad_{}'.format(rad), doi2)
        self.add_scalars('degree_of_interest3_rad_{}'.format(rad), doi3)
        print('Topography calculated')
        

# class meshviewer:
#     def __init__(self, shape = (1, 1), bg = 'black'):
#         self.actors = {}
#         self.p = pv.Plotter(shape = shape, notebook = False, window_size = (750, 500))
#         self.p.background_color = bg
#         if bg == 'black':
#             self.text_color = 'white'
#         else:
#             self.text_color = 'black'
#         self.p.link_views()
#         self.shift = None
#     def add_actor(self, newmesh, loc = (0, 0), percentiles = [10, 90], scalar_name = 'depth_map', 
#                   opacity = 1, colormap = 'viridis', color = None, clipping = None, inverting = False, edges = False, 
#                   ambient = 0.0, diffuse = 1., specular = 0., specular_power = 90, smooth_shading = None,
#                   add_light = False):
#         self.p.subplot(*loc)
#         actor = topograph()
#         actor.merge_with(newmesh)
#         scalars = actor.point_data[scalar_name].copy()
#         if self.shift is None:
#             self.shift = actor.points.mean(axis = 0) 
#         actor.points = actor.points - self.shift
#         if percentiles is not None:
#             low, high = percentiles
#             lowq = np.percentile(scalars, low)
#             highq = np.percentile(scalars, high)
#             scalars[scalars <= lowq] = lowq
#             scalars[scalars >= highq] = highq
#         actor.point_data[scalar_name] = scalars
#         actor.sas(scalar_name)
#         if clipping is not None:
#             actor = actor.clip(clipping, invert = inverting)
#         self.p.add_mesh(actor, cmap = colormap, color = color, show_edges = edges,
#                         lighting = True, opacity = opacity, ambient = ambient, 
#                         diffuse = diffuse, specular = specular, specular_power = specular_power,
#                         smooth_shading = smooth_shading, scalar_bar_args = {'color': self.text_color})
#         if add_light:
#             light = pv.Light(position = (1.0, 1.0, 1.0),
#                              focal_point = (0, 0, 0),
#                              color = [1, 1.0, 0.9843, 1],  # Color temp. 5400 K
#                              intensity = 0.9)
#             self.p.add_light(light)
#         self.p.camera.focal_point = (0, 0, 0)
#         # self.p.camera.direction((0, 0, -1))
#         # self.p.camera.distance = 5
#         # self.p.camera.zoom(0.7)
#         # self.actors[scalar_name] = actor
#         return actor
#     def show(self):
#         self.p.show()
#         self.shift = None
    









