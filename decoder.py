# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 23:55:59 2018

@author: DELL
"""
import pandas as pd
import numpy as np
import cv2 
import sys

quantilisation_table_512 = pd.read_csv('512.csv')
quantilisation_table_512 = quantilisation_table_512.values
quantilisation_table_1024 = pd.read_csv('1024.csv')
quantilisation_table_1024 = quantilisation_table_1024.values

def transform_table(N):
  transform_table = np.zeros((N,N))
  transform_table[0, :] = 1 * np.sqrt(1/N)  
  for i in range(1, N):  
       for j in range(N):  
            transform_table[i, j] = np.cos(np.pi * i * (2*j+1) / (2 * N )  
  ) * np.sqrt(2 / N ) 
  return transform_table

transform_table_512 = transform_table(512)
transform_table_1024 = transform_table(1024)
print('transform table created')

#----------------------decode--------------
#input your encode file
def decode(encode_data = '1.1.13'):
  quantilisation_table =  np.array([])
  transform_table = np.array([])
  N= 0
  dir_name = 'encode_data/'
  encode_data_path = dir_name +encode_data
  try:
    decompressed_read = pd.read_csv(encode_data_path,compression = 'gzip',encoding = 'utf-8')
    decompressed_read_img = decompressed_read.values    
    if int(encode_data_path[-4])<3:
      N = 512
      quantilisation_table =  quantilisation_table_512
      transform_table = transform_table_512
    else:
      N = 1024
      quantilisation_table = quantilisation_table_1024
      transform_table = transform_table_1024
  except:
    print('Please enter the correct file path:')
    sys.exit(1)

  
  columns = N
  rows = N
   
  max_row_zero_index,max_column_zero_index = decompressed_read_img.shape
  max_column_zero_index =max_column_zero_index -1
  max_row_zero_index =max_column_zero_index-1
  print(max_column_zero_index)
  print(max_row_zero_index)
  
  #截取还原
  decompressed_img = np.zeros((columns, rows)) 
  for i in range(max_column_zero_index):
    for j in range(max_row_zero_index):
      decompressed_img[j,i] = decompressed_read_img[j,i]
  
  
  #反转IDCT decode 后的图像
  print(quantilisation_table)
  img_decode = decompressed_img*quantilisation_table
  img_decode = np.dot(np.transpose(transform_table) , img_decode)  
  img_decode = np.dot(img_decode, transform_table)
  #转换成整数
  img_decode = np.around(img_decode,decimals = 0)
  
  #图片输出展示
  output_file = 'single_decode/'
  img_name= encode_data_path.split('/')[1]
  out_path = output_file+img_name+'.tiff'
  cv2.imwrite(out_path,img_decode)

if __name__ == '__main__':
  encode_data = input('Please enter the file path you need to decode: ')
  decode(encode_data)
  print('Decode Done!')