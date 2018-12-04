#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:57:46 2018

@author: hayashi
"""


import os
import numpy as np
import cv2
from PIL import Image

size = 160
step = 45

path = "/home/sora/project"
for d_num in [1,2,3,4]:
    print("Stert %d"%d_num)
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    lists = os.listdir(path + "/data/ips/dataset")
    os.chdir(path + "/ips/PSPnet2/data/ips/dataset")
    
    #testdata
    f = open("test_data%d.txt" %(d_num))
    lines = f.readlines()
    f.close()
    
    #改行を取り除く
    for q in range(len(lines)):
        lines[q] = lines[q].strip()
    for line in lines:
        path1 = os.path.join(path,line)
        img = Image.open(path1)
        width,hight = img.size
        for y in range(0, hight-size+1, step):
            for x in range(0, width-size+1, step):
                I = np.array(img.crop((x, y, x+size, y+size)))
                I.resize(160,160,3)
                X_test.append(I)
                    
    #testmask
    f = open("test_mask%d.txt" %(d_num))
    lines = f.readlines()
    f.close()
    
    #改行を取り除く
    for r in range(len(lines)):
        lines[r] = lines[r].strip()
    
    for line in lines:
        path2 = os.path.join(path,line)
        im2 = cv2.imread(path2)
        for y in range(0, hight-size+1, step):
            for x in range(0, width-size+1, step):
                c_im2 = im2[y:size+y, x:size+x]
                #c_im2.resize(473,473,3)
                imim = c_im2[:,:,0]+c_im2[:,:,1]+c_im2[:,:,2]
                c_im2[:,:,0]=(imim==255)*c_im2[:,:,0]
                c_im2[:,:,1]=(imim==255)*c_im2[:,:,1]
                c_im2[:,:,2]=(imim==255)*c_im2[:,:,2]
                c_im2 = c_im2/255
                Y_test.append(c_im2)
                
    #traindata
    f = open("train_data%d.txt" %(d_num))
    lines = f.readlines()
    f.close()
    
    #改行を取り除く
    for q in range(len(lines)):
        lines[q] = lines[q].strip()
        
    for line in lines:
        path1 = os.path.join(path,line)
        img = Image.open(path1)
        width,hight = img.size
        for y in range(0, hight-size+1, step):
            for x in range(0, width-size+1, step):
                I = np.array(img.crop((x, y, x+size, y+size)))
                I.resize(160,160,3)
                X_train.append(I)
                    
    #trainmask
    f = open("train_mask%d.txt" %(d_num))
    lines = f.readlines()
    f.close()
    
    #改行を取り除く
    for r in range(len(lines)):
        lines[r] = lines[r].strip()
    
    for line in lines:
        path2 = os.path.join(path,line)
        im2 = cv2.imread(path2)
        for y in range(0, hight-size+1, step):
            for x in range(0, width-size+1, step):
                c_im2 = im2[y:size+y, x:size+x]
                #c_im2.resize(473,473,3)
                imim = c_im2[:,:,0]+c_im2[:,:,1]+c_im2[:,:,2]
                c_im2[:,:,0]=(imim==255)*c_im2[:,:,0]
                c_im2[:,:,1]=(imim==255)*c_im2[:,:,1]
                c_im2[:,:,2]=(imim==255)*c_im2[:,:,2]
                c_im2 = c_im2/255
                Y_train.append(c_im2)
                
                
    X_test = np.asarray(X_test) / 255.
    X_train = np.asarray(X_train) / 255.
    
    
    os.chdir(path + "/ips/PSPnet2/X_crop/%d/%d"%(size,d_num))
       
    np.save('X_test.npy',X_test)
    np.save('X_train.npy',X_train)
    X_test = []
    X_train = []
    
    Y_test = np.asarray(Y_test) / 255.
    Y_train = np.asarray(Y_train) / 255.
    
    os.chdir("/home/sora/project/ips/PSPnet2/Y_crop/%d/%d"%(size,d_num))
       
    np.save('Y_test.npy',Y_test)
    np.save('Y_train.npy',Y_train)
    Y_test = []
    Y_train = []
    
