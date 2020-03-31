# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:11:31 2020

@author: Elizabeth
"""

import numpy as np
import matplotlib.pyplot as plt
#from sklearn import preprocessing
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix
#from sklearn import metrics
import skimage
from skimage import io, color, morphology, filters, exposure                    
import os

#%% Lectura de imagenes 
path_sanos = 'sanos/'
path_leucemia = 'leucemia/'

for name_sanos in os.listdir(path_sanos):
    img_sanos = io.imread(path_sanos + name_sanos)
    img_grayS = color.rgb2gray(img_sanos)
    img_grayS = skimage.img_as_ubyte(img_grayS) #cambio de uint8 a la escala de (0-255)
    img_grayS = morphology.dilation(img_grayS).astype(np.uint8)
    img_grayS = filters.sobel(img_grayS)
    
    
for name_leu in os.listdir(path_leucemia):
    img_leucemia = io.imread(path_leucemia + name_leu)
    img_grayL = color.rgb2gray(img_leucemia)
    img_grayL = skimage.img_as_ubyte(img_grayL) #cambio de uint8 a la escala de (0-255)
    img_grayL = morphology.dilation(img_grayL).astype(np.uint8)
    img_grayL = filters.sobel(img_grayL)
    

img_gray = np.concatenate((img_grayL,img_grayS))
    
#%% PAC

pcaObj = PCA(2)
PcaTrain = pcaObj.fit_transform(img_gray)#Matriz de covarianza 
PcaTrain = PcaTrain.T 

plt.figure(figsize = (10,24))
plt.scatter(PcaTrain[0][:480],PcaTrain[1][:480], color = 'm', label = 'Celulas con Leucemia')
plt.scatter(PcaTrain[0][480:],PcaTrain[1][480:], color = 'c', label = 'Celulas Sanas')
plt.xlabel('PCATrain1')
plt.ylabel('PCATrain2')
plt.legend()
plt.title('Análisi de Componentes Principates (PCA) en la matriz de entrenamiento')
plt.show()

#%% KernelLPCA
KlpcaObj = KernelPCA(n_components=2,kernel='poly')
KlpcaTrain = KlpcaObj.fit_transform(img_gray) 
KlpcaTrain = KlpcaTrain.T 

plt.figure(figsize = (10,24))
plt.scatter(KlpcaTrain[0][:480],KlpcaTrain[1][:480], color = 'm', label = 'Celulas con Leucemia')
plt.scatter(KlpcaTrain[0][480:],KlpcaTrain[1][480:], color = 'c', label = 'Celulas Sanas')
plt.xlabel('KlPCATrain1')
plt.ylabel('KlPCATrain2')
plt.legend()
plt.title('Análisi de compinentes principales  por kernel (KlPCA) en la matriz de entrenamiento')
plt.show()

#%% LDA 

#Vectores propios (Etiquetas)
target1 = np.zeros(img_grayL.shape[0])# vector para la clase 1 leucemia
target2 = np.ones(img_grayS.shape[0])# vector para la clase 2 sanos
target = np.concatenate((target1,target2))
target = target.T

ldaObj = LDA(n_components=2)
LdaTrain = ldaObj.fit(img_gray,target)
LdaTrain = ldaObj.transform(img_gray)
LdaTrain = LdaTrain.T


plt.figure(figsize = (10,24))
plt.scatter(LdaTrain[0][:240],LdaTrain[0][240:480], color = 'm', label = 'Celulas con Leucemia')
plt.scatter(LdaTrain[0][480:720],LdaTrain[0][720:], color = 'c', label = 'Celulas Sanas')
plt.xlabel('LCATrain1')
plt.ylabel('LCATrain2')
plt.legend()
plt.title('Análisi Discriminante Lineal (LDA) en la matriz de entrenamiento')
plt.show()