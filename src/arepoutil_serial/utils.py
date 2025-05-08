"""
This script was developed by Martin Bourne and is based on previous code developed by Debora Sijacki. 
"""

import numpy as np
import math
num_cores = 1 
from ctypes import *
import os
dir = os.path.abspath(os.path.dirname(__file__))
so_file = os.path.join(dir, "./imageUtils.so")
imageUtils = CDLL(so_file)

def cutCubeIndices(pos, x1, y1 = None, z1 = None, x2 = None, y2 = None, z2 = None):
    y1 = x1 if y1 == None else y1
    z1 = x1 if z1 == None else z1
    x2 = -x1 if x2 == None else x2
    y2 = -y1 if y2 == None else y2
    z2 = -z1 if z2 == None else z2
    x, y, z = pos.T
    Xmin = min(x1,x2)
    Xmax = max(x1,x2)
    Ymin = min(y1,y2)
    Ymax = max(y1,y2)
    Zmin = min(z1,z2)
    Zmax = max(z1,z2)
    inc = np.where((x <= Xmax) & (x >= Xmin) & (y <= Ymax) & (y >= Ymin) & (z <= Zmax) & (z >= Zmin))[0]
    return inc

def getImageArray(content):
    core, Xpixels, Ypixels, mass, xa, ya, za, ha, halfDepth, width, height, quant = content
    nPixels = Xpixels * Ypixels
    image = np.zeros(nPixels)
    cImage = (c_double * nPixels)(*image)
    npart = len(mass)
    print(len(quant), len(mass))
    if len(quant) == 0:
        imageUtils.getMassArray.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_double, c_double, c_double, c_int, POINTER(c_double)]
        imageUtils.getMassArray(Xpixels, Ypixels, (c_double * npart)(*mass), (c_double * npart)(*xa), (c_double * npart)(*ya), (c_double * npart)(*za), (c_double * npart)(*ha), width, height, halfDepth, npart, cImage)
        image = cImage[:]
        return image
    else:
        weight = np.zeros(nPixels)
        cWeight = (c_double * nPixels)(*weight)
        imageUtils.getWeightedQuantityArray.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), c_double, c_double, c_double, c_int, POINTER(c_double), POINTER(c_double)]
        imageUtils.getWeightedQuantityArray(Xpixels, Ypixels, (c_double * npart)(*mass), (c_double * npart)(*quant), (c_double * npart)(*xa), (c_double * npart)(*ya), (c_double * npart)(*za), (c_double * npart)(*ha), width, height, halfDepth, npart, cImage, cWeight)
        image = cImage[:]
        weight = cWeight[:]
        return [image, weight]

def prepProjection(mass, pos, width, height, Xpixels, Ypixels):
    npart = len(mass)
    xa, ya, za = pos.T
    pixelSizeX = width / Xpixels
    pixelSizeY = height / Ypixels
    # shifting the x and y coordinates so that the origin is in the centre of the plot
    xa += width / 2
    ya += height / 2
    nPerCore = math.ceil(npart / num_cores)
    return [npart, xa, ya, za, pixelSizeX, pixelSizeY, nPerCore]

def doProjection(pos, mass, hsml, Xpixels, Ypixels, width, height, depth, quant = []):
    npart, xa, ya, za, pixelSizeX, pixelSizeY, nPerCore = prepProjection(mass, pos, width, height, Xpixels, Ypixels)
    imageContent = []
    start = 0
    end = nPerCore
    for core in range(0, num_cores):
        imageContent.append([
            core,
            Xpixels,
            Ypixels,
            mass[start:end],
            xa[start:end],
            ya[start:end],
            za[start:end],
            hsml[start:end],
            depth / 2,
            width,
            height,
            quant[start:end]
        ])
        start += nPerCore
        end += nPerCore
        if end > npart:
            end = npart
    image_array = [getImageArray(imageContent[0])]
    if len(quant) == 0:
        return np.reshape(np.sum(image_array,0), [Xpixels, Ypixels]).T / (pixelSizeX * pixelSizeY)
    else:
        w, q = np.swapaxes(image_array,0,1)
        return np.reshape(np.sum(q,0).T, [Xpixels, Ypixels]).T / np.reshape(np.sum(w,0), [Xpixels, Ypixels]).T

def doProjectQuantity(pos, mass, quant, hsml, Xpixels, Ypixels, width, height, depth):
    return doProjection(pos, mass, hsml, Xpixels, Ypixels, width, height, depth, quant)