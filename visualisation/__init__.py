# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:29:10 2019

@author: ozdemir
"""

""" Here I implemented a simple image viewer which uses matplotlib's imshow to view 2D images inline. 
    If 3D images are passed as input, they are also plotted in a slicewise fashion. """


import matplotlib.pyplot as plt
# import numpy as np


def view2d(*imgs, slice_no = 0):
    if len(imgs) == 1:
        plt.figure()
        plt.imshow(imgs[0], interpolation='none')
        text = 'image: {}, slice: {}'.format(0, slice_no)
        plt.text(5, -4, text)
        return()
    else:
        items = [item for item in imgs]
        num = len(items)
        fig, axes = plt.subplots(1, num, figsize = (num * 4, num * 3), sharey = True)
        for i, (ax, item) in enumerate(zip(axes, items)):
            ax.imshow(item, interpolation = 'none')
            text = 'image: {}, slice: {}'.format(i, slice_no)
            ax.text(5, -4, text)
        plt.show()
        

def view3d(*volumes):
    range_ = len(volumes[0])
    for i in range(range_):
        items = [item[i] for item in volumes]
        view2d(*items, slice_no = i)
    

def view(*items):
    if items[0].ndim == 2:
        view2d(*items)
    elif items[0].ndim == 3:
        view3d(*items)
        
    






