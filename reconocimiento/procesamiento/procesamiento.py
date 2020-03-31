# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:24:21 2020

@author: Elizabeth
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import exposure, io, color, morphology, filters
from scipy import ndimage

#%%
img = io.imread('leucemia.jpg')
img_gray = color.rgb2gray(img)
img_gray = skimage.img_as_ubyte(img_gray)

#%% Matematica

img_ecu = exposure.equalize_adapthist(img_gray)
img_gray_ecu = img_gray + img_ecu
img_gray_ecu_rest = img_gray - img_ecu
img_total = img_gray_ecu - img_gray_ecu_rest

img_fil = filters.sobel(img_total)

plt.figure
plt.imshow(img_fil, cmap=plt.cm.gray)
plt.show

#img_hist = exposure.histogram(img_gray, nbins=256)
#
#plt.figure(figsize=(8,16))
#plt.plot(img_hist[1], img_hist[0], lw=2)
#plt.xlabel('Intensidad de color')
#plt.xlim(0,255)
#plt.ylabel('Numero de pixeles')
#plt.title('Histograma')
#
#plt.figure(figsize=(10,25))
#plt.subplot(2,3,1)
#plt.imshow(img)
#plt.title('Original')
#plt.subplot(2,3,2)
#plt.title('Escala de grices')
#plt.imshow(img_gray, cmap=plt.cm.gray)
#plt.subplot(2,3,3)
#plt.imshow(img_ecu, cmap=plt.cm.gray)
#plt.title('Ecualizada')
#plt.subplot(2,3,4)
#plt.imshow(img_gray_ecu, cmap=plt.cm.gray)
#plt.title('Suma gris y ecualizada')
#plt.subplot(2,3,5)
#plt.title('Resta gris y ecualizada')
#plt.imshow(img_gray_ecu_rest, cmap=plt.cm.gray)
#plt.subplot(2,3,6)
#plt.imshow(img_total, cmap=plt.cm.gray)
#plt.title('Imagen resultante')

#%% Funciones

img_dila = morphology.dilation(img_gray).astype(np.uint8)
img_ero = morphology.erosion(img_gray).astype(np.uint8)
img_open = morphology.opening(img_gray).astype(np.uint8)
img_clos = morphology.closing(img_gray).astype(np.uint8)

img_fil_d = filters.sobel(img_dila)

plt.figure
plt.imshow(img_fil_d, cmap=plt.cm.gray)
plt.show

#plt.figure(figsize=(10,25))
#plt.subplot(2,3,1)
#plt.imshow(img)
#plt.title('Original')
#plt.subplot(2,3,2)
#plt.title('Escala de grices')
#plt.imshow(img_gray, cmap=plt.cm.gray)
#plt.subplot(2,3,3)
#plt.title('Dilatacion')
#plt.imshow(img_dila, cmap=plt.cm.gray)
#plt.subplot(2,3,4)
#plt.title('Erosion')
#plt.imshow(img_ero, cmap=plt.cm.gray)
#plt.subplot(2,3,5)
#plt.title('Opening')
#plt.imshow(img_open, cmap=plt.cm.gray)
#plt.subplot(2,3,6)
#plt.title('Closing')
#plt.imshow(img_clos, cmap=plt.cm.gray)

#%%

lab = np.where((img_gray>140)&(img_gray<220))#Indentificamos los pixeles que corresponden a la hoja
int_AVG = ndimage.measurements.mean(img_clos[lab[0],lab[1]])#Calculamos intensidad de color promedio
con_AVG = ndimage.measurements.standard_deviation(img_clos[lab[0],lab[1]])#Calculamos el contraste promedio

print('La Intensidad promedio es: ', int_AVG)
print('El Contraste promedio es: ', con_AVG)
