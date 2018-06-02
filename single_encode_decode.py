# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:23:13 2018

@author: DELL
"""

import math
import numpy as np
import cv2  
import pandas as pd
import os
from PSNR import PSNR

#df_output_desription = pd.DataFrame()

def create_quantilisation_table (N):
  
  #创建指数型增加的量化矩阵
  power = math.log2(N) #梯度
  division_index = np.logspace(1,power,power,base = 2,dtype = 'int') -1 #梯度列表
  
  quantilization_table = np.zeros((N,N))#创建量化矩阵
  adjusted_index = -2 #调整最后梯度的索引
  last_index = 0 #梯度起始值
  
  #量化矩阵赋值
  for q_index in division_index[:adjusted_index]: 
    quantilization_table[last_index:q_index,:] = q_index
    quantilization_table[:,last_index:q_index] = q_index 
    last_index  = q_index
    
  #量化矩阵调整赋值
  for q_index in division_index[adjusted_index:]: 
    quantilization_table[last_index:q_index,:] = division_index[adjusted_index-1]
    quantilization_table[:,last_index:q_index] = division_index[adjusted_index-1] 
    last_index  = q_index
     
  quantilization_table[division_index[-1],:] = division_index[adjusted_index-1]
  quantilization_table[:,division_index[-1]:] = division_index[adjusted_index-1]
  
  return quantilization_table

quantilisation_table_512 = create_quantilisation_table(512)
quantilisation_table_1024 = create_quantilisation_table(1024)
print('quantilisation table created')
#dic = {'img_name':[],'compression_ratio':[],'bit_rate':[],'PSNR':[]}

def create_transform_table(N):
  transform_table = np.zeros((N,N))
  transform_table[0, :] = 1 * np.sqrt(1/N)  
  for i in range(1, N):  
       for j in range(N):  
            transform_table[i, j] = np.cos(np.pi * i * (2*j+1) / (2 * N )  
  ) * np.sqrt(2 / N )
  
  return transform_table
transform_table_512 = create_transform_table(512)
transform_table_1024 = create_transform_table(1024)

print('transform table created')

def encode_decode(img_location='1.3.03.tiff'):   
  # 读取相片矩阵
  img_name = img_location.split('.tiff')[0]
#  dic['img_name'].append(img_name)
  print(img_name)
  img_location = 'Textures/'+img_location
  img = cv2.imread(img_location, 0)   
  img1 = img.astype('float')
  #print(img1)  
  
  #创建离散余弦变换矩阵
  transform_table = np.zeros(img.shape)
  output_table = np.zeros(img.shape)  
  
  #获取需要压缩图片的长和宽
  columns, rows = img.shape
  N = rows
  if N == 512:
    transform_table = transform_table_512
  else:
    transform_table = transform_table_1024
    
  '''
  N = rows
  
  #离散余弦变换矩阵输入
  transform_table[0, :] = 1 * np.sqrt(1/N)  
  for i in range(1, columns):  
       for j in range(rows):  
            transform_table[i, j] = np.cos(np.pi * i * (2*j+1) / (2 * N )  
  ) * np.sqrt(2 / N )  
  '''
  #print(transform_table)
  #对源图进行离散余弦变换
  output_table = np.dot(transform_table, img1)  
  output_table = np.dot(output_table, np.transpose(transform_table))
  
  
  quantilization_table = np.zeros(img.shape)
  if N == 512:
    quantilization_table = quantilisation_table_512
  else:
    quantilization_table = quantilisation_table_1024
    

  #乘以量化矩阵
  img_encode = output_table/quantilization_table 
  img_encode = np.around(img_encode,decimals = 0)
  
  
  #零截取
  row_zero_index_list  = []
  row_zero_index = 0
  for pixel in reversed(img_encode[:,0]):#行列， 第0列 , 遍历每一行的第一个 row    
    if pixel == 0:
      row_zero_index_list.append(row_zero_index)   
    else:
      break
    row_zero_index += 1
    
  if len(row_zero_index_list):
    pass
  else:
    row_zero_index_list.append(0)
  max_row_zero_index = rows- max(row_zero_index_list)-1
  #print(max_row_zero_index)
  
  #------------------------------------
  column_zero_index_list  = []
  column_zero_index = 0
  for pixel in reversed(img_encode[0,:]):#行列， 第0列 
    if pixel == 0:
      column_zero_index_list.append(column_zero_index)   
    else:
      break
    column_zero_index += 1
    
  if len(column_zero_index_list):
    pass
  else:
    column_zero_index_list.append(0)
  
  max_column_zero_index = columns- max(column_zero_index_list)-1
  #print(max_column_zero_index)
  
  #------------------------------------
  #零数目的pixel 截掉
  compressed_img = img_encode[:max_row_zero_index,:max_column_zero_index ]

  # 输出压缩的encode图片，并且gzip 进行压缩
  encode_data_path = 'encode_data/'+img_name
  df_compress_img = pd.DataFrame(compressed_img)
  df_compress_img.to_csv(encode_data_path,
                         compression = 'gzip',
                         index = False,
                         encoding = 'utf-8')
  
  #compression_ratio
  fsize_encode = os.path.getsize(encode_data_path)
  fsize_original = os.path.getsize(img_location)
  compression_ratio = round(fsize_original/fsize_encode,2)
  #dic['compression_ratio'].append(compression_ratio)
  
  #bitrate
  bit_rate = round(fsize_encode/(N*N),2)
  #dic['bit_rate'].append(bit_rate)
  
  
  #print(dic)
  
  #-------------------------------------encode done!-------------------------------
  
  
  #-----------------------------------decode--------------------------------------
  #  读取encod_img
  decompressed_read = pd.read_csv(encode_data_path,compression = 'gzip',encoding = 'utf-8')
  decompressed_read_img = decompressed_read.values
  
  #截取还原
  decompressed_img = np.zeros((columns, rows)) 
  for i in range(max_column_zero_index):
    for j in range(max_row_zero_index):
      decompressed_img[j,i] = decompressed_read_img[j,i]
   
  #反转IDCT decode 后的图像
  img_decode = decompressed_img*quantilization_table
  img_decode = np.dot(np.transpose(transform_table) , img_decode)  
  img_decode = np.dot(img_decode, transform_table)
  #转换成整数
  img_decode = np.around(img_decode,decimals = 0)
  
  #图片输出展示
  output_file = 'all_decode_img/'
  out_path = output_file+img_name+'.tiff'
  cv2.imwrite(out_path,img_decode)
  
  #PSRP
  psnr = PSNR(img_name+'.tiff')
  #dic['PSNR'].append(psnr)
  
  #img_decode_read = cv2.imread(out_path,0)
  #print(img_decode)
  #print(img_decode_read)
  return img_name,psnr,bit_rate,compression_ratio

if __name__ == '__main__':
  #Default decode image is 1.3.03.tiff
  img = input('please enter the img name you need to encode:')
  encode_decode(img)
  print('Encode & Decode Done!')
