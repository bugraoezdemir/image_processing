# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:36:12 2020

@author: bugra
"""


import numpy as np
import SimpleITK as sitk
    

""" This module is aimed to provide an easy access to the SimpleITK filters for direct use with numpy arrays. 
    Each of the adapted functions below accepts and returns numpy arrays. The descriptions
    for the individual filters were also copied from the corresponding SimpleITK filter documents. """


def adaptive_histogram_equalisation(img, alpha = 0.5, beta = 0.5, rad = 2): 
    """ The parameter alpha controls how much the filter acts like the classical histogram equalization method 
        (alpha=0) to how much the filter acts like an unsharp mask (alpha=1). The parameter beta controls how much 
        the filter acts like an unsharp mask (beta=0) to much the filter acts like pass through (beta=1, with alpha=1).
        The parameter window controls the size of the region over which local statistics are calculated.
        """
    im = sitk.GetImageFromArray(img)
    filtered = sitk.AdaptiveHistogramEqualizationImageFilter()
    filtered.SetAlpha(alpha)
    filtered.SetBeta(beta)
    filtered.SetRadius(rad)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

def bilateral(img, domainsig = 1, rangesig = 5, nrgs = 20):
    """ This filter uses bilateral filtering to blur an image using both domain and range "neighborhoods". 
        Pixels that are close to a pixel in the image domain and similar to a pixel in the image range are used 
        to calculate the filtered value. Two gaussian kernels (one in the image domain and one in the image range) 
        are used to smooth the image. The result is an image that is smoothed in homogeneous regions yet has edges preserved. 
        The result is similar to anisotropic diffusion but the implementation in non-iterative. Another benefit to bilateral 
        filtering is that any distance metric can be used for kernel smoothing the image range.
        """
    im = sitk.GetImageFromArray(img)
    filtered = sitk.BilateralImageFilter()
    filtered.SetDomainSigma(domainsig)
    filtered.SetRangeSigma(rangesig)
    filtered.SetNumberOfRangeGaussianSamples(nrgs)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr   

def curvature_flow(img, timestep = 0.125, iterations = 5):
    img = sitk.GetImageFromArray(img)
    imgSmooth = sitk.CurvatureFlow(image1 = img, timeStep = 0.125, numberOfIterations = 5)
    arr = sitk.GetArrayFromImage(imgSmooth)
    return arr   

def curvature_anisotropic_diffusion(img, conductance = 1, interval = 2, iterations = 10, timestep = 0.2):
    im = sitk.GetImageFromArray(img)
    filtered = sitk.CurvatureAnisotropicDiffusionImageFilter()
    filtered.SetConductanceParameter(conductance)
    filtered.SetConductanceScalingUpdateInterval(interval)
    filtered.SetNumberOfIterations(iterations)
    filtered.SetTimeStep(timestep)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr 

def gradient_anisotropic_diffusion(img, conductance = 1, interval = 2, iterations = 10, timestep = 0.2):
    im = sitk.GetImageFromArray(img)
    filtered = sitk.GradientAnisotropicDiffusionImageFilter()
    filtered.SetConductanceParameter(conductance)
    filtered.SetConductanceScalingUpdateInterval(interval)
    filtered.SetNumberOfIterations(iterations)
    filtered.SetTimeStep(timestep)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr   

def expnegative(img):
    """This is a quick and effective denoising filter."""
    im = sitk.GetImageFromArray(img)
    filtered = sitk.ExpNegativeImageFilter()
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr  

def fastrank(img, rad = 2, rank = 0.2):
    """This is a quick and effective smoothing filter."""
    im = sitk.GetImageFromArray(img)
    filtered = sitk.FastApproximateRankImageFilter()
    filtered.SetRadius(rad)
    filtered.SetRank(rank)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr  

def shapelyclose (img, full_connectivity = False, radius = 5, kernel = 1, preserve_intensities = False):
    """ This filter is similar to the morphological closing, but contrary to the mophological closing, the closing 
        by reconstruction preserves the shape of the components. The closing by reconstruction of an image "f" is defined as:
        ClosingByReconstruction(f) = ErosionByReconstruction(f, Dilation(f)). Closing by reconstruction not only preserves 
        structures preserved by the dilation, but also levels raises the contrast of the darkest regions. If PreserveIntensities 
        is on, a subsequent reconstruction by dilation using a marker image that is the original image for all unaffected pixels.
        kernel no points to indice of this list: Annulus=sitkAnnulus, Ball=sitkBall, Box=sitkBox, Cross=sitkCross
        """
    im = sitk.GetImageFromArray(img)
    filtered = sitk.ClosingByReconstructionImageFilter()
    filtered.SetFullyConnected(full_connectivity)
    filtered.SetKernelRadius(radius)
    filtered.SetKernelType(kernel)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr  

def greyclose(img, seed = [3, 3, 3]):
    im = sitk.GetImageFromArray(img)
    filtered = sitk.GrayscaleConnectedClosingImageFilter()
    filtered.SetSeed(seed)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

def iterative_greyclose(img, seed = [1, 2, 2], iterations = 3):
    im = img.copy()
    for i in range(iterations):
        im = greyclose(im, seed)
    return im

def binary_shapelyclose(binary, full_connectivity = False, radius = [2, 2, 2], kernel = 1):
    binary = binary.astype(int)
    im = sitk.GetImageFromArray(binary)
    filtered = sitk.BinaryClosingByReconstructionImageFilter()
    filtered.SetKernelRadius(radius)
    filtered.SetKernelType(kernel)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)   
    return arr  

def binary_median(binary, radius = [2,2,1], kernel = 1):
    binary = binary.astype(int)
    im = sitk.GetImageFromArray(binary)
    filtered = sitk.BinaryMedianImageFilter()
    filtered.SetRadius(radius)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)   
    return arr  

def binary_minmax_curvature_flow(binary, timestep = 0.1, rad = 2, thresh = 0, iterations = 40):
    binary = binary.astype(np.float32)
    im = sitk.GetImageFromArray(binary)
    filtered = sitk.BinaryMinMaxCurvatureFlowImageFilter()
    filtered.SetTimeStep(timestep)
    filtered.SetThreshold(thresh)
    filtered.SetNumberOfIterations(iterations)
    filtered.SetStencilRadius(rad)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)   
    return arr  

def voting_binary(binary, radius = [1, 1, 1], birth = 10, survival = 11):
    binary = binary.astype(int)
    im = sitk.GetImageFromArray(binary)
    filtered = sitk.VotingBinaryImageFilter()
    filtered.SetRadius(radius)
    filtered.SetBirthThreshold(birth)
    filtered.SetSurvivalThreshold(survival)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)   
    return arr  

def voting_binary_iterative_holefilling(img, rad = [3, 3, 3], iterations = 5, majority_thresh = 1):
    # img = adseg.hystero(x,1,1).astype(int)
    im = sitk.GetImageFromArray(img)
    filtered = sitk.VotingBinaryIterativeHoleFillingImageFilter()
    filtered.SetRadius(rad)
    filtered.SetMaximumNumberOfIterations(iterations)
    filtered.SetMajorityThreshold(majority_thresh)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

def hconcave(img, h = 0.2):
    """The output from this filter must be inverted"""
    im = sitk.GetImageFromArray(img)
    filtered = sitk.HConcaveImageFilter()
    filtered.SetHeight(h)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

def hconvex(img,h = 0.99):
    """This denoising filter might be especially good for cluster segmentation.
    h must be slightly lower than 1. If h is too low, it loses a lot of information but
    this might help with the cluster segmentation"""
    im = sitk.GetImageFromArray(img)
    filtered = sitk.HConvexImageFilter()
    filtered.SetHeight(h)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

def hmaxima(img,h = 0.2):
    """Keep h slightly higher than 0"""
    im = sitk.GetImageFromArray(img)
    filtered = sitk.HMaximaImageFilter()
    filtered.SetHeight(h)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

def hminima(img,h = 0.02):
    """HMinimaImageFilter suppresses local minima that are less than h intensity units below the (local) 
    background. This has the effect of smoothing over the "low" parts of the noise in the image without 
    smoothing over large changes in intensity (region boundaries). See the HMaximaImageFilter to suppress 
    he local maxima whose height is less than h intensity units above the (local) background. If original 
    image is subtracted from the output of HMinimaImageFilter , the signicant "valleys" in the image can 
    be identified. This is what the HConcaveImageFilter provides.
    This filter uses the GrayscaleGeodesicErodeImageFilter . It provides its own input as the "mask" input 
    to the geodesic dilation. The "marker" image for the geodesic dilation is the input image plus the height 
    parameter h.
    """
    im = sitk.GetImageFromArray(img)
    filtered = sitk.HMinimaImageFilter()
    filtered.SetHeight(h)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

def N4BiasFieldCorrection(img, thresh = 0.001, fwhm = 0.15, iterations = [4, 50], wiener_noise = 0.01):
    im = sitk.GetImageFromArray(img)
    filtered = sitk.N4BiasFieldCorrectionImageFilter()
    filtered.SetConvergenceThreshold(thresh)
    filtered.SetMaximumNumberOfIterations(iterations)
    filtered.SetBiasFieldFullWidthAtHalfMaximum(fwhm)
    filtered.SetWienerFilterNoise(wiener_noise)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

def minmax_curvature_flow(img,timestep = 0.1,rad = 2,iterations = 40):
    im = sitk.GetImageFromArray(img)
    filtered = sitk.MinMaxCurvatureFlowImageFilter()
    filtered.SetTimeStep(timestep)
    filtered.SetNumberOfIterations(iterations)
    filtered.SetStencilRadius(rad)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

def laplacian_sharpening(img):
    im = sitk.GetImageFromArray(img)
    filtered = sitk.LaplacianSharpeningImageFilter()
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

def rankfilter(img, rad = [1,2,2], rank = 0.1):
    im = sitk.GetImageFromArray(img)
    filtered = sitk.RankImageFilter()
    filtered.SetRadius(rad)
    filtered.SetRank(rank)
    result = filtered.Execute(im)
    arr = sitk.GetArrayFromImage(result)
    return arr

#################################################################################################################
#################################################################################################################


############################################# Deconvolution filters #############################################
#################################################################################################################

def landweber_deconvolution(img, kernel_shape = (3, 3, 3), alpha = 0.5, iterations = 30, normalise = True):
    """Projected Landweber Deconvolution method using a flat kernel."""
    kernel = np.ones(kernel_shape)/np.prod(kernel_shape)
    kernel = kernel.astype(np.float32)
    img = img.astype(np.float32)
    im = sitk.GetImageFromArray(img)
    stkpsf = sitk.GetImageFromArray(kernel)
    filtered = sitk.ProjectedLandweberDeconvolutionImageFilter()
    filtered.SetAlpha(alpha)
    filtered.SetNumberOfIterations(iterations)
    filtered.SetNormalize(True)
    result = filtered.Execute(im,stkpsf)
    arr = sitk.GetArrayFromImage(result)
    return arr

def rlu_deconvolution(img, kernel_shape = (3, 3, 3), iterations = 20, normalise = True):
    """Projected Richardson Lucy Deconvolution method."""
    kernel = np.ones(kernel_shape) / np.prod(kernel_shape)
    kernel = kernel.astype(np.float32)
    img = img.astype(np.float32)
    im = sitk.GetImageFromArray(img)
    stkpsf = sitk.GetImageFromArray(kernel)
    filtered = sitk.RichardsonLucyDeconvolutionImageFilter()
    filtered.SetNumberOfIterations(iterations)
    filtered.SetNormalize(normalise)
    result = filtered.Execute(im,stkpsf)
    arr = sitk.GetArrayFromImage(result)
    return arr

def tikhonov_deconvolution(img, kernel_shape = (3, 3, 3),reg = 0.1,normalise = True):
    """A non-iterative deconvolution method."""
    kernel = np.ones(kernel_shape)/np.prod(kernel_shape)
    kernel = kernel.astype(np.float32)
    img = img.astype(np.float32)
    im = sitk.GetImageFromArray(img)
    stkpsf = sitk.GetImageFromArray(kernel)
    filtered = sitk.TikhonovDeconvolutionImageFilter()
    filtered.SetNormalize(True)
    filtered.SetRegularizationConstant(reg)
    result = filtered.Execute(im,stkpsf)
    arr = sitk.GetArrayFromImage(result)
    return arr

def wiener_deconvolution(img, kernel_shape = (3, 3, 3), var = 0.5, normalise = False):
    """ A non-iterative deconvolution method. """
    kernel = np.ones(kernel_shape) / np.prod(kernel_shape)
    kernel = kernel.astype(np.float32)
    img = img.astype(np.float32)
    im=sitk.GetImageFromArray(img)
    stkpsf = sitk.GetImageFromArray(kernel)
    filtered = sitk.WienerDeconvolutionImageFilter()
    filtered.SetNormalize(True)
    filtered.SetNoiseVariance(var)
    result = filtered.Execute(im, stkpsf)
    arr = sitk.GetArrayFromImage(result)
    return arr

#################################################################################################################
#################################################################################################################


def correlate(img, kernel_shape, mask = None):
    if np.isscalar(kernel_shape):
        kernel_shape = np.ones([kernel_shape] * img.ndim)
    if not hasattr(kernel_shape, 'nonzero'):
        kernel = np.ones(kernel_shape) / np.prod(kernel_shape)
    else:
        kernel = kernel_shape.astype(np.float32)
    im = sitk.GetImageFromArray(img.astype(np.float32))
    kernel = sitk.GetImageFromArray(kernel)
    if mask is None:
        mask = np.ones_like(img)
    mask = mask.astype(np.float32)
    mask = sitk.GetImageFromArray(mask)
    result = sitk.NormalizedCorrelation(im, maskImage = mask, templateImage = kernel)
    arr = sitk.GetArrayFromImage(result)
    return arr     
