# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:45:27 2018

@author: DELL
"""
import time
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from single_encode_decode import encode_decode
start = time.clock()
# 读取所有的相片文件
rootdir = os.path.dirname(os.path.abspath(__file__))
document_name = '\Textures'
img_dir = rootdir + document_name
img_name_list = [item for item in os.listdir(img_dir) if item.endswith('.tiff')]


dic = {'img_name':[],'compression_ratio':[],'bit_rate':[],'PSNR':[]}

# 多线程 压缩解压
with ThreadPoolExecutor(max_workers = 20) as executor:
  task_list = executor.map(encode_decode,img_name_list)
  for img_index in task_list:#一个图像的 img_name,psnr,bit_rate,compression_ratio
    dic['img_name'].append(img_index[0])
    dic['PSNR'].append(img_index[1])
    dic['bit_rate'].append(img_index[2])
    dic['compression_ratio'].append(img_index[3])

df_measure_index = pd.DataFrame(dic)
df_measure_index.to_excel('measure_index.xls',index = False)
end = time.clock()
print('Use Time:{:.2f}'.format(end-start))
    
    
    
  
  
    
