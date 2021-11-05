# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 12:57:17 2020

@author: bugra
"""

""" This module contains several basic vector algebra operations that I frequently use.
    """

import numpy as np
from ...utils import convenience as cnv

def get_uvecs(data, ndim = None):
    """ Converts an iterable of vectors into unit vectors. 

        Parameters:
        -----------
        data: iterable or array 
            A list, tuple or array of vectors.
        ndim: int
            Can be left as None. Should be specified only if the input
            vectors are unstructured (a 1-dimensional point list). 

        Returns:
        --------
        uvecs: array of floats
            Unit vectors rendered from the input vectors. 
        """
    if ndim is None:
        if np.isscalar(data[0]):
            ndim = 3
            print('A flat array is given. Assumed to be 3D-dimensional space.')
        else:
            ndim = data.shape[1]
    vectors = cnv.ensure2dpointlist(data, ndim = ndim)
    normed = np.linalg.norm(vectors, axis = 1).reshape(len(vectors), 1)
    uvecs = vectors / normed
    return uvecs

def to_origin(data):
    """ Translates a list of points or an object so that the centroid sits at the origin.

        Parameters:
        -----------
        data: iterable or array
            Either a 3D binary object or a 2D iterable representing coordinates of an object.

        Returns:
        --------
        vectors_centered: array 
            Array of vectors representing the translated coordinates of the object. 
        """
    coords = cnv.check_dataform(data, return_pointlist = True)
    centroid = coords.mean(axis = 0)
    vectors_centered = coords - centroid.reshape(1, -1)
    return vectors_centered

def to_uvecs(data):
    """ First translates points to set centroid to the origin and then converts the translated 
        coordinate vectors into unit vectors. 
        
        Parameters: 
        -----------
        data: iterable or array
            Either a 3D binary object or a 2D iterable representing coordinates.
        
        Returns:
        --------
        uvecs: array 
            An array of vectors representing the translated unit-vector coordinates of the object.
        """
    vectors_centered = to_origin(data)
    return get_uvecs(vectors_centered)

def get_angles(u, v, in_radians = False, clipping = (-1, 1)):
    """ This function calculates the angle or angles between two sets of vectors and returns an angle matrix.
        The vertical axis of this matrix shows the indices of the vector set u and the horizontal axis shows the 
        indices of the vector set v. 

        Parameters:
        -----------
        u: iterable or array
            first set of vector coordinates.
        v: iterable or array
            second set of vector coordinates.
        in_radians: boolean 
            if True, the angle values are given in radians, else in degrees. 
        clipping: tuple of two
            clips the cosines within the range specified.

        Returns:
        --------
        angles: array
            A 2D numpy array with the angle values between each combination of the vectors from vector sets u and v.
            The rows of the array represent the indices of the vector set u and the columns represent the indices of the 
            vector set v. 
        """
    u = get_uvecs(u)
    v = get_uvecs(v)
    lowc, highc = clipping
    angles_in_radian = np.arccos(np.clip(np.dot(u, v.T), lowc, highc))
    if in_radians == True:
        return angles_in_radian
    else:
        angles = np.nan_to_num(angles_in_radian)
        return np.rad2deg(angles)

def get_angles_indexwise(u, v, in_radians = False, clipping = (-1, 1)):
    """ This function calculates the angle or angles between two sets of vectors in an index-wise manner. So the angle is 
        calculated between a vector at index n of u and a vector at index n of v. Here u and v must have equal lengths (so
        they have equal number of vectors). 
        
        Parameters:
        -----------
        u: iterable or array
            first set of vector coordinates.
        v: iterable or array
            second set of vector coordinates.
        in_radians: boolean 
            if True, the angle values are given in radians, else in degrees. 
        clipping: tuple of two
            clips the cosines within the range specified.

        Returns:
        --------
        angles: array
            A 2D numpy array with the angle values between the vectors in u and the corresponding vectors in v. 
        """
    u = get_uvecs(u)
    v = get_uvecs(v)
    vectoral_dot_product = np.sum(u * v, axis = 1)
    angles_in_radian = np.arccos(np.clip(vectoral_dot_product, *clipping))
    if in_radians == True:
        return angles_in_radian
    else:
        angles = np.nan_to_num(angles_in_radian)
        return np.rad2deg(angles)

def line_projection(u, v): 
    """ This function projects points in u onto the vector v. 

        Parameters:
        -----------
        u: iterable or array
            First set of vector coordinates.
        v: iterable or array
            A single vector, to which the points in u will be projected. 

        Returns:
        --------
        projected: array
            A numpy array with the projected points. This array the same length as u. 
        """ 
    if not hasattr(u, 'nonzero'):
        u = np.array(u)
    if not hasattr(v, 'nonzero'):
        v = np.array(v)
    if u.ndim == 1:
        u = u.reshape(1, len(u))
    if v.ndim == 1:
        v = v.reshape(1, len(v))
    assert v.shape[0] == 1, 'v must be a single vector.'
    uv = np.dot(u, v.T)
    vv = np.dot(v, v.T)
    projected = (uv / vv) * v
    return projected

def plane_projection(u, v): 
    """ This function projects points in u onto the surface defined by v. 

        Parameters:
        -----------
        u: iterable or array
            A 2D iterable of vector coordinates
        v: iterable or array
            A 2D iterable of vector coordinates. 

        Returns:
        --------
        Projected: array 
            A numpy array with the projected points. This array the same length as u. 
        """ 
    return (u - line_projection(u, v))

def cut_plane(u, v, level): 
    """ This function projects points in u onto the surface defined by v to create a cutting plane.
        Then the cutting plane is translated along the vector v by a distance specified with 'level'.

        Parameters:
        -----------
        u: iterable or array
            A 2D iterable of vector coordinates
        v: iterable or array
            A 2D iterable of vector coordinates. 
        level: The level along v, at which the cutting plane is calculated. It is the 
            fraction of the full length along v.
        
        Returns:
        --------
        Projected: array 
            A numpy array with the projected points. This array the same length as u. 
        """ 
    if not hasattr(u, 'nonzero'):
        u = np.array(u)
    if not hasattr(v, 'nonzero'):
        v = np.array(v)
    if u.ndim == 1:
        u = u.reshape(1, len(u))
    if v.ndim == 1:
        v = v.reshape(1, len(v))
    assert v.shape[0] == 1, 'v must be a single vector.'
    proj_line = line_projection(u, v)
    midpoint = proj_line.mean(axis = 0)
    return (u - proj_line + 2 * midpoint * level)

def cut_with_plane(vol, v, dist_fract):
    """ This function calculates a cutting plane in 3D and applies it to the
        input volume 'vol'. The cutting plane is calculated perpendicular to the
        vector 'v' and with a distance that is a fraction of the full length along v. 

        Parameters:
        -----------
        vol: array
            A 3D numpy array
        v: iterable or array
            A 2D iterable of vector coordinates. 
        level: float 
            The level along v, at which the cutting plane is applied. It is the 
            fraction of the full length along v.
        
        Returns:
        --------
        masked: array 
            Input volume 'vol' masked with the cutting plane. 
            A numpy array with the same shape as 'vol'.
        """ 
    cnv.cp_3d(vol) # make sure that data is a 3D volume.
    coords = np.argwhere(np.ones_like(vol))
    proj = cut_plane(coords, v, dist_fract)
    plane = cnv.coords_to_array(proj, vol)
    return plane * vol

def get_vectors_by_angle (pts0, pts1, angle = 90, tol = 10):  
    """ This function searches two sets of vectors for pairs of vectors that make a particular angle.  

        Parameters:
        -----------
        pts0: iterable or array 
            First iterable of vectors to search  
        pts1: iterable or array
            Second iterable of vectors to search
        tol: float 
            Maximum tolerable deviation from the input angle degree to accept that two vectors make the input angle.

        Returns:
        --------
        vectors: tuple of arrays
            The indices of the vectors found by the search
        """   
    pts0 = cnv.check_dataform(pts0)
    pts1 = cnv.check_dataform(pts1)
    angles = get_angles(pts0, pts1)
    mask = (angles >= angle - tol) & (angles <= angle + tol)
    args = np.argwhere(mask)
    return args

def draw_line_3D(pt0, pt1):
    """ This function draws a 3D straight line from point pt0 to point pt1.  

        Parameters:
        -----------
        pt0: iterable of size 3 
            An iterable of 3 integers, indicating the coordinates of the first point
        pt0: iterable of size 3 
            An iterable of 3 integers, indicating the coordinates of the second point

        Returns:
        --------            
        line: tuple of size 3 
            A tuple of 3x 1D arrays, indicating coordinates of the drawn line. 
        """ 
    if not hasattr(pt0, 'nonzero'):
        pt0 = np.array(pt0)
    if pt0.ndim == 1:
        pt0 = pt0.reshape(1, len(pt0))
    if not hasattr(pt1, 'nonzero'):
        pt1 = np.array(pt1)
    if pt1.ndim == 1:
        pt1 = pt1.reshape(1, len(pt1))
    pt00, pt01, pt02 = tuple(pt0.T)
    pt10, pt11, pt12 = tuple(pt1.T)
    length = np.linalg.norm(pt1 - pt0, axis = 1).flatten().astype(int)[0]
    newpts0 = np.linspace(pt00, pt10, length * 3)
    newpts1 = np.linspace(pt01, pt11, length * 3)
    newpts3 = np.linspace(pt02, pt12, length * 3)
    line = np.hstack((newpts0, newpts1, newpts3)).astype(int)    
    return tuple(line.T)

