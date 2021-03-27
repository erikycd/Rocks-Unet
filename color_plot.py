# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:17:18 2021

@author: Erik
"""

import cv2 as cv

def mask_color_img(img, mask_1, color_1, mask_2, color_2, alpha):
    '''
    img: cv2 image
    mask: bool or np.where
    color: BGR triplet [_, _, _]. Default: [0, 255, 255] is yellow.
    alpha: float [0, 1]. 

    Ref: http://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    '''
    out = img.copy()
    img_layer = img.copy()
    img_layer[mask_1] = color_1
    out = cv.addWeighted(img_layer, alpha, out, 1 - alpha, 0, out)

    out[mask_2] = color_2
    out2 = cv.addWeighted(out, alpha, out, 1 - alpha, 0, out)

    return out2