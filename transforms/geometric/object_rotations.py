# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:05:03 2019

@author: ozdemir
"""


""" This module contains several functions related to rotation of 3D volumes.  
    """

import numpy as np
from scipy import spatial as spt
# from math import sin, cos
from ...utils import convenience as cnv
from . import vector_algebra as va

def get_rotation_matrix(theta, axis):
    """ This function calculates a rotation matrix that can rotate an object around a given 
        axis in 3D through a given angle theta. The axis must be one of
        the three main axes. It cannot be an arbitrary vector.

        Parameters:
        -----------
        theta: scalar 
            The angle of rotation in degrees).
        axis: int 
            Axis, around which the rotation is performed (ijk style).

        Returns:
        --------
        R_theta: array
            Rotation matrix with shape (3, 3).
        """
    theta = np.radians(theta)
    if axis == 0:
        R_theta = np.array([[1, 0, 0],
                            [0, np.cos(theta), -np.sin(theta)],
                            [0, np.sin(theta), np.cos(theta)]])
    elif axis == 1:
        R_theta = np.array([[np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)]])
    else:
        R_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
    return R_theta 


def get_rotation_matrices(angles, axis):
    """ This function calculates multiple 3D rotation matrices that correspond to 
        mmultiple different angles provided by the 'angles' parameter. Note that 
        the axis here must be one of the three main axes. It cannot be an arbitrary vector. 
        
        Parameters:
        -----------
        angles: iterable of size 3
            The angles of rotation in degrees.
        axis: int
            Axis, around which rotations are performed (in 'ij' order).

        Returns:
        --------
        R_theta: 3D numpy array. Rotation matrices, concatenated at 2nd dimension,
                corresponding to each angle in the 'angles' input.
        """
    angles = np.radians(angles)
    nulls = np.zeros(len(angles))
    eins = np.ones(len(angles))
    if axis == 0:
        R_angles = np.array([[eins, nulls, nulls],
                            [nulls, np.cos(angles), -np.sin(angles)],
                            [nulls, np.sin(angles), np.cos(angles)]])
    elif axis == 1:
        R_angles = np.array([[np.cos(angles), nulls, np.sin(angles)],
                            [nulls, eins, nulls],
                            [-np.sin(angles), nulls, np.cos(angles)]])
    else:
        R_angles = np.array([[np.cos(angles), -np.sin(angles), nulls],
                            [np.sin(angles), np.cos(angles), nulls],
                            [nulls, nulls, eins]])
    return R_angles


def angleaxis2rotmat (theta, axis):
    ''' 3D Rotation matrix of angle 'theta' around 'vector'. This function is better than
        the function 'get_rotation_matrix' since this one can calculate rotations around
        any arbitrary axis specified as 'axis'

        Note: This function is based on the following wikipedia article: 
                http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
                
        Parameters:
        -----------
        theta: scalar 
            Angle of rotation in degrees.                
        vector: iterable 
            An iterable of size 3. Specifies the axis for rotation. 

        Returns
        -------
        rotation_matrix: array
            A (3x3) array representing the rotation matrix
        '''
    theta = np.deg2rad(theta)
    x, y, z = axis
    c = np.cos(theta); s = np.sin(theta); C = 1-c
    xs = x * s;   ys = y * s;   zs = z * s
    xC = x * C;   yC = y * C;   zC = z * C
    xyC = x * yC; yzC = y * zC; zxC = z * xC
    return np.array([[ x *xC + c,   xyC - zs,   zxC + ys ],
                     [ xyC + zs,   y * yC + c,   yzC - xs ],
                     [ zxC - ys,   yzC + xs,   z * zC + c ]])


def rotate (obj, theta, axis, scale = (1, 1, 1), return_img = False): 
    """ A function to rotate given 3D binary object. A rotation matrix is calculated and used 
            to rotate the input object.       
            
        Note: This function is currently incomplete. An interpolation step will be added soon.
        
        Parameters:
        ----------
        obj: array-like
            The object to be rotated. Either a coordinate list or the actual object. 
        theta: scalar  
            The angle of rotation in degrees. 
        axis: iterable 
            Vector specifying the axis around which to perform the rotations. 
        scale: iterable
            Iterable specifying the spatial scale of the input object.
        return_img: boolean. 
            If True, the rotated object is transformed into an image to be returned. If False,
            the rotated object is returned as a 2D array specifying the voxel coordinate. 
            
        Returns:
        -------
        rotated: array
            Either a pointset representing coordinates of the rotated data 
            or 2D/3D numpy array representing the actual rotated object.
        """
    pts = cnv.check_dataform(obj)
    ndim = pts.shape[1]
    axis = cnv.ensure2dpointlist(axis, ndim)
    scale = cnv.ensure2dpointlist(scale, ndim)
    assert axis.shape[1] == ndim, 'The "axis" parameter must be a vector of length equal to the object dimensionality'
    scalars = obj[tuple(pts.astype(int).T)]
    current_loc = pts.mean(axis = 0)
    pts = pts - current_loc
    pts = pts * scale
    axis = va.get_uvecs(axis).ravel()
    rot_mat = angleaxis2rotmat(theta, axis)
    rotated = np.dot(rot_mat, pts.T).T
    rotated = rotated / scale
    backtranslated = rotated + current_loc
    if return_img == True:        
        return cnv.coords_to_array(backtranslated, obj, scalars)
    else:
        return rotated   



def rotor(obj, theta = 180, axis = (0, 1, 0), scale = (1, 1, 1), return_img = False):
    """ A function to rotate given 3D binary object. A rotation matrix is calculated and used 
            to rotate the input object.       
        
    
        Note: This is basically the same as 'rotate' above, except  that this one uses
            scipy's rotation function instead of dot product.
        
        Parameters:
        ----------
        obj: array-like
            The object to be rotated. Either a coordinate list or the actual object. 
        theta: scalar  
            The angle of rotation in degrees. 
        axis: iterable 
            Vector specifying the axis around which to perform the rotations. 
        scale: iterable
            Iterable specifying the spatial scale of the input object.
        return_img: boolean. 
            If True, the rotated object is transformed into an image to be returned. If False,
            the rotated object is returned as a 2D array specifying the voxel coordinate. 
            
        Returns:
        -------
        rotated: array
            Either a pointset representing coordinates of the rotated data 
            or 2D/3D numpy array representing the actual rotated object.
        """
    pts = cnv.check_dataform(obj)
    ndim = pts.shape[1]
    axis = cnv.ensure2dpointlist(axis, ndim)
    scale = cnv.ensure2dpointlist(scale, ndim)
    assert axis.shape[1] == ndim, 'The "axis" parameter must be a vector of length equal to the object dimensionality'
    scalars = obj[tuple(pts.astype(int).T)]
    current_loc = pts.mean(axis = 0)
    pts = pts - current_loc
    pts = pts * scale
    axis = va.get_uvecs(axis).ravel()
    rot_mat = angleaxis2rotmat(theta, axis)
    rotated = spt.transform.Rotation.from_matrix(rot_mat).apply(pts)
    rotated = rotated / scale
    backtranslated = rotated + current_loc
    if return_img == True:        
        return cnv.coords_to_array(backtranslated, obj, scalars)
    else:
        return rotated   







