# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 09:44:33 2018

@author: DELL
"""

import cv2
import numpy as np
import math


def PSNR(img_name = '1.3.03.tiff'):
  #print(img_name)
  original_img = 'Textures/'+img_name
  decode_img = 'all_decode_img/'+img_name
  #print(original_img)
  #print(decode_img)
  
  img_original = cv2.imread(original_img , 0)
  img_encode = cv2.imread(decode_img, 0)
  #print(img_original)
  #print(img_encode)
  difference  = img_original - img_encode
  sqr = difference * difference
  MSE = np.mean(sqr)
  PSNR = round(10*math.log10((255*255)/MSE),2)
  return PSNR
  
if __name__ == '__main__':
  #有对应 已经 解压 decode 的图片 才能运行
  print(PSNR('1.3.03.tiff'))
  
  
  
