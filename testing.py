# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:21:35 2018

@author: DELL
"""

s = ''
try:
  s = None
  if s is None:
    print("s 是空对象")
    raise NameError     #如果引发NameError异常，后面的代码将不能执行
    print(len(s))  #这句不会执行，但是后面的except还是会走到
except NameError: #对NameError 错误进行报错
  print ("空对象没有长度")
print('hhefs')