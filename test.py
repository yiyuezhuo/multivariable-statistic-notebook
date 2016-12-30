# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 17:36:31 2016

@author: yiyuezhuo
"""

import pandas as pd
import scipy.stats as stats

with open('homework2-3.txt',encoding='utf8') as f:
    s = f.read()
    
sl = [ss for ss in s if len(s)>0]
sll=[]
for i in range(10):
    sll.append(sl[i*6:i*6+6])
    
df= pd.DataFrame(sll[1:])
df.columns= sll[0]
# 此时还有些数据清理问题