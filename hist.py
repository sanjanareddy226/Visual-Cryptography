#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:33:29 2020

@author: sanjanasrinivasareddy
"""

import matplotlib.pyplot as plt
import cv2 

# Histogram
img = plt.imread('outputKN.png',0)
histr = cv2.calcHist([img],[0],None,[256],[0,256]) 
plt.plot(histr) 

img2 = plt.imread('secret1.jpg',0)
histr = cv2.calcHist([img2],[0],None,[256],[0,256]) 
plt.plot(histr)

#mse

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err

import numpy as np
a = cv2.imread('secret1.jpg')
b = cv2.imread('outputKN.png')
res = cv2.resize(b, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
mse(a,res)

#npcrv

from PIL import Image
import numpy as np
def rateofchange(height,width,pixel1,pixel2,matrix,i):

    for y in range(0,height):
        for x in range(0,width):
            #print(x,y)
            if pixel1[x,y][i] == pixel2[x,y][i]:
                matrix[x,y]=0
            else:
                matrix[x,y]=1
    return matrix
def sumofpixel(height,width,pixel1,pixel2,ematrix,i):
    matrix=rateofchange(height,width,pixel1,pixel2,ematrix,i)
    psum=0
    for y in range(0,height):
        for x in range(0,width):
            psum=matrix[x,y]+psum
    return psum
def npcrv(loc1,loc2):
    c1 = Image.open(loc1)
    c2 = Image.open(loc2)
    width, height = c1.size
    pixel1 = c1.load()
    pixel2 = c2.load()
    ematrix = np.empty([width, height])
    per=(((sumofpixel(height,width,pixel1,pixel2,ematrix,0)/(height*width))*100)+((sumofpixel(height,width,pixel1,pixel2,ematrix,1)/(height*width))*100)+((sumofpixel(height,width,pixel1,pixel2,ematrix,2)/(height*width))*100))/3
    return per

print(npcrv("secret1.jpg","outputKN.png"))



#corr coeff
from PIL import Image
from math import sqrt
value_of_x=0
value_of_y=0
def co1(color_index_of_rgb,height,width,pixels):
    value=0
    for pixel_coordinate_of_y in range(0, height):
        for pixel_coordinate_of_x in range(0, width):
            if pixel_coordinate_of_x+1==width:
                break
            value=pixels[pixel_coordinate_of_x,pixel_coordinate_of_y][color_index_of_rgb]*pixels[pixel_coordinate_of_x+1,pixel_coordinate_of_y][color_index_of_rgb]+value


    return value*height*width

def co2(color_index_of_rgb,height,width,pixels):
   global value_of_y
   global value_of_x
   for pixel_coordinate_of_y in range(0, height):
        for pixel_coordinate_of_x in range(0, width):
            if pixel_coordinate_of_x+1==width:
                break
            value_of_x=pixels[pixel_coordinate_of_x,pixel_coordinate_of_y][color_index_of_rgb]+value_of_x
            value_of_y=pixels[pixel_coordinate_of_x+1,pixel_coordinate_of_y][color_index_of_rgb]+value_of_y

   return value_of_x*value_of_y


def co3(color_index_of_rgb,height,width,pixels):
    value=0
    for pixel_coordinate_of_y in range(0, height):
        for pixel_coordinate_of_x in range(0, width):
            value=(pixels[pixel_coordinate_of_x,pixel_coordinate_of_y][color_index_of_rgb])**2 +value

    xy=(value*height*width)-(value_of_x**2)
    return  xy

def co4(color_index_of_rgb,height,width,pixels):
    value=0
    for pixel_coordinate_of_y in range(0, height):
        for pixel_coordinate_of_x in range(0, width):
            if pixel_coordinate_of_x+1==width:
                break
            value=(pixels[pixel_coordinate_of_x+1,pixel_coordinate_of_y][color_index_of_rgb]**2)+value

    xy=(value*height*width)-(value_of_y**2)
    return xy

def corr_of_rgb(loc):
    global value_of_y
    global value_of_x
    photo = Image.open(loc)
    pixels = photo.load()
    width, height = photo.size
    value_of_y = 0
    value_of_x = 0
    r=((co1(0,height,width,pixels)-co2(0,height,width,pixels)) / sqrt(co3(0,height,width,pixels)*co4(0,height,width,pixels)))
    value_of_y=0
    value_of_x=0
    g=((co1(1,height,width,pixels) - co2(1,height,width,pixels)) / sqrt(co3(1,height,width,pixels) * co4(1,height,width,pixels)))
    value_of_x=0
    value_of_y=0
    b=((co1(2,height,width,pixels) - co2(2,height,width,pixels)) / sqrt(co3(2,height,width,pixels) * co4(2,height,width,pixels)))

    return ((r+g+b)/3)
corr_of_rgb("outputKN.png")

#coeff graph
from PIL import Image
def coplot_vertical(loc):
    image = Image.open(loc)
    pixels = image.load()
    list_of_pixels_of_x = []
    list_of_pixels_of_y = []

    width, height = image.size
    for pixel_coordinate_of_y in range(0, 50):
        for pixel_coordinate_of_x in range(0, 50):
            list_of_pixels_of_x.append(pixels[pixel_coordinate_of_y,pixel_coordinate_of_x][0])
            if pixel_coordinate_of_y + 1 == height:
                list_of_pixels_of_y.pop()
                break
            else:
                list_of_pixels_of_y.append(pixels[pixel_coordinate_of_y , pixel_coordinate_of_x+1][0])

    plt.scatter(list_of_pixels_of_x, list_of_pixels_of_y, label='Pixel', color='k', s=2, edgecolors='r')
    plt.xlabel('Pixel value on location(x,y)')
    plt.ylabel('Pixel value on location(x+1,y)')
    plt.title("correlation coefficient graph")
    plt.legend()
    plt.show()

coplot_vertical("outputKN.png")

def coplot_horizontal(loc):
    image = Image.open(loc)

    pixels=image.load()
    list_of_pixels_of_x=[]
    list_of_pixels_of_y=[]
    width,height=image.size
    for pixel_coordinate_of_y in range(0,50):
        for pixel_coordinate_of_x in range(0,50):
            list_of_pixels_of_x.append(pixels[pixel_coordinate_of_x,pixel_coordinate_of_y][0])
            if pixel_coordinate_of_x+1 == width:
                list_of_pixels_of_x.pop()
                break
            else:
                list_of_pixels_of_y.append(pixels[pixel_coordinate_of_x+1,pixel_coordinate_of_y][0])

    plt.scatter(list_of_pixels_of_x,list_of_pixels_of_y,  label='Pixel',color='k',s=2,edgecolors='r')
    plt.xlabel('Pixel value on location(x,y)')
    plt.ylabel('Pixel value on location(x+1,y)')
    plt.title("correlation coefficient graph")
    plt.legend()
coplot_horizontal("outputKN.png")
