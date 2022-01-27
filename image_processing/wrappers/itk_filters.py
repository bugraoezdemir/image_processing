# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 00:08:18 2021

@author: bugra
"""

import itk
import numpy as np


########################################## Here some itk functions that are useful for my tasks #####################################
#####################################################################################################################################

def im2array(img):
    """ Converts an itk image type to a numpy array. 
        Parameters:
        -----------
        img: itkImage
        
        Returns:
            A numpy array converted from itk image. 
        """
    array = itk.GetArrayFromImage(img)
    return array    
    
def array2im(array):
    """ Converts an numpy array type to an itk image. 
        img: array

        Returns:
            An itk image converted from a numpy array. 
        """
    img = itk.GetImageFromArray(array)
    return img
    
def getslice(img, slcno, as_np = False):
    """ Get a single slice from a 3D itk image. 

        Parameters:
        -----------
        slcno: scalar, the index of the slice to select from the itk stack.
        as_np: scalar, if true, the slice is returned as a 2D numpy array
        s: scalar, if nonzero, the returned array is illustrated. 

        Returns:
        --------
        a 2D itk image or numpy array. 
        """
    array = im2array(img)
    slc = array[slcno]
    if as_np == True:
        return slc
    else:
        imslc = array2im(slc)
        return imslc

def caster(img, dim=2, firsttype=itk.UC, secondtype=itk.F):
    """ Cast an itk image to a desired datatype. 

        Parameters:
        -----------
        dim: number of dimensions of the input / output.
        firsttype: the data type of the input. 
        secondtype: the data type of the output. 

        Returns:
        --------
        The itk image casted to desired datatype. 
        """
    imgtype0 = itk.Image[firsttype,dim]
    imgtype1 = itk.Image[secondtype,dim]
    casted = itk.CastImageFilter[imgtype0,imgtype1].New()
    casted.SetInput(img)
    return casted.GetOutput()

def simple_threshold(img, lower, higher):
    """ Threshold a 3D itk image between a lower and a higher threshold values. 

        Parameters:
        -----------
        lower: lower threshold
        higher: higher threshold
        s: scalar, if nonzero, the returned array is illustrated. 

        Returns:
        --------
        The thresholded image. 
        """
    imgtype = itk.Image[itk.F, 3]
    thresholder=itk.BinaryThresholdImageFilter[imgtype, imgtype].New()
    thresholder.SetLowerThreshold(lower)
    thresholder.SetUpperThreshold(higher)
    thresholder.SetOutsideValue(0)
    thresholder.SetInsideValue(1)
    thresholder.SetInput(img)
    thresholded = thresholder.GetOutput()
    thresholded = im2array(thresholded)
    return thresholded

def otsu_threshold(img, as_np = True):
    """ Threshold a 3D itk image using otsu thresholding method. 

        Parameters:
        -----------
        s: scalar, if nonzero, the returned array is illustrated. 

        Returns:
        --------
        The thresholded image. 
        """
    imgtype1 = itk.Image[itk.F, 3]
    imgtype2 = itk.Image[itk.SS, 3]
    thresholder = itk.OtsuThresholdImageFilter[imgtype1, imgtype2].New()
    thresholder.SetOutsideValue(1)
    thresholder.SetInsideValue(0)
    thresholder.SetInput(img)
    thresholded = thresholder.GetOutput()
    if as_np:    
        thresholded = im2array(thresholded)
    return thresholded    

def laplacian_gaussian(img, sigma = 1):
    """ Transform an image with the laplacian of gaussian image filter. 

        Parameters:
        -----------
        sigma: scalar, if nonzero, the returned array is illustrated. 
        s: scalar, if nonzero, the returned array is illustrated. 

        Returns:
        --------
        The filtered image. 
        """
    imgtypeinput = itk.Image[itk.F, 3]
    imgtypeoutput = itk.Image[itk.F, 3]
    lapfilter = itk.LaplacianRecursiveGaussianImageFilter[imgtypeinput,imgtypeinput].New()
    lapfilter.SetInput(img)
    lapfilter.SetSigma(sigma)
    lapfilterout = lapfilter.GetOutput()
    rescaler = itk.RescaleIntensityImageFilter[imgtypeinput, imgtypeoutput].New()
    rescaler.SetInput(lapfilterout)
    outputPixelTypeMinimum = itk.NumericTraits[itk.UC].min()
    outputPixelTypeMaximum = itk.NumericTraits[itk.UC].max()
    rescaler.SetOutputMinimum(outputPixelTypeMinimum)
    rescaler.SetOutputMaximum(outputPixelTypeMaximum)
    toreturn = rescaler.GetOutput()
    toreturn = im2array(toreturn)
    return toreturn

def canny(img, variance, lower, higher, s=0):
    """ Transform an image with the canny image filter. 
    
        Parameters:
        -----------
        variance: scalar, must be non-positive, controls the spatial spreading 
            of the filtered signal
        lower: the lower threshold in the hysteresis thresholding step of the canny filter 
        higher: the higher threshold in the hysteresis thresholding step of the canny filter
        s: scalar, if nonzero, the returned array is illustrated. 
        
        Returns:
        --------
        The filtered image. 
        """
    variance = float(variance)
    lower = float(lower)
    higher = float(higher)
    imgtypeinput = itk.Image[itk.F, 3]
    imgtypeoutput = itk.Image[itk.F, 3]
    cannyFilter = itk.CannyEdgeDetectionImageFilter[imgtypeinput, imgtypeinput].New()
    cannyFilter.SetInput(img)
    cannyFilter.SetVariance(variance)
    cannyFilter.SetLowerThreshold(lower)
    cannyFilter.SetUpperThreshold(higher)
    cannyFilterout = cannyFilter.GetOutput()
    rescaler = itk.RescaleIntensityImageFilter[imgtypeinput, imgtypeoutput].New()
    rescaler.SetInput(cannyFilterout)
    outputPixelTypeMinimum = itk.NumericTraits[itk.F].min()
    outputPixelTypeMaximum = itk.NumericTraits[itk.F].max()
    rescaler.SetOutputMinimum(outputPixelTypeMinimum)
    rescaler.SetOutputMaximum(outputPixelTypeMaximum)
    toreturn = rescaler.GetOutput()
    toreturn = im2array(toreturn)
    return toreturn


############################################ Here some itk based vesselness enhancement filters #####################################
#####################################################################################################################################

def enhance_vesselness(imgset, sig = 1, alpha1 = 0.5, alpha2 = 2):
    itkimg = itk.GetImageFromArray(imgset.astype(np.float32))
    input_image = itkimg
    hessian_image = itk.hessian_recursive_gaussian_image_filter(input_image, sigma = sig)
    vesselness_filter = itk.Hessian3DToVesselnessMeasureImageFilter[itk.ctype('float')].New()
    vesselness_filter.SetInput(hessian_image)
    vesselness_filter.SetAlpha1(alpha1)
    vesselness_filter.SetAlpha2(alpha2)
    out = vesselness_filter.GetOutput()
    outarr = itk.GetArrayFromImage(out)
    return outarr

def enhance_vesselness_multiscale(imgset, sigmin = 0.01, sigmax = 3, signum = 24, alpha = 3,
                          beta = 0.5, gamma = 0.005, scale_objectness = False):
    input_image = itk.GetImageFromArray(imgset.astype(np.float32))
    ImageType = type(input_image)
    Dimension = input_image.GetImageDimension()
    HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
    HessianImageType = itk.Image[HessianPixelType, Dimension]
    objectness_filter = itk.HessianToObjectnessMeasureImageFilter[HessianImageType, ImageType].New()
    objectness_filter.SetBrightObject(True)
    objectness_filter.SetScaleObjectnessMeasure(scale_objectness)
    objectness_filter.SetAlpha(alpha)
    objectness_filter.SetBeta(beta)
    objectness_filter.SetGamma(gamma)
    multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[ImageType, HessianImageType, ImageType].New()
    multi_scale_filter.SetInput(input_image)
    multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
    multi_scale_filter.SetSigmaStepMethodToLogarithmic()
    multi_scale_filter.SetSigmaMinimum(sigmin)
    multi_scale_filter.SetSigmaMaximum(sigmax)
    multi_scale_filter.SetNumberOfSigmaSteps(signum)
    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
    rescale_filter.SetInput(multi_scale_filter)
    output=rescale_filter.GetOutput()
    output_array = itk.GetArrayFromImage(output)
    return output_array

def enhance_vesselness_2D(img, sigmin = 0.01, sigmax = 3, signum = 24,
                          alpha = 3, beta = 0.5, gamma = 0.005, scale_objectness = False):
    newim = img.copy()
    for i, slc in enumerate(img):
        filtered = enhance_vesselness_multiscale(slc, sigmin, sigmax, signum, 
                                                 alpha, beta, gamma, scale_objectness)
        newim[i] = filtered
    return newim


