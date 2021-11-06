# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:36:12 2020

@author: bugra
"""



try:
    from image_processing.wrappers import __init__
    wrappers_exists = True
except:
    wrappers_exists = False
    raise Exception('The simple_itk_filters subpackage is currently not available\
    because the "wrappers" package is not yet included in the pip installation.\
    This issue will soon be solved as there will be a separate installation\
    recipe for the "wrappers" package.')


if wrappers_exists:
    from image_processing.transforms.photometric.local_filtering.simple_itk_filters import *