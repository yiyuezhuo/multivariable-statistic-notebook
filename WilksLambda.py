# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 23:24:22 2016

@author: yiyuezhuo
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

path = 'C:\\Users\\yiyuezhuo\\Desktop\\多元统计\\例2-1long.xls'
df = pd.read_excel(path)

dfs = df[['净资产收益率','总资产报酬率','资产负债率','销售增长率']]
gb = dfs.groupby(df['行业'])
X = []
for _,sdf in gb:
    sdf.index = range(len(sdf))
    X.append(np.array(sdf))

r = len(X)
d = len(X[0][0])
E = np.zeros((d,d))
for k in range(r):
    for j in range(len(X[k])):
        term = X[k][j] - X[k].mean(axis=0)
        E += np.outer(term,term)

W = np.zeros((d,d))
X_bar = dfs.mean(axis=0)
for k in range(r):
    for j in range(len(X[k])):
        term = X[k][j] - X_bar
        W += np.outer(term,term)

B = W - E

def SSCP(dfs, group_index):
    # 离均差平方和与离均差积和矩阵 
    # Sum Of Squares And Cross-Products Matrix, SSCP 
    gb = dfs.groupby(group_index)
    X = []
    for _,sdf in gb:
        sdf.index = range(len(sdf))
        X.append(np.array(sdf))
    
    r = len(X)
    d = len(X[0][0])
    E = np.zeros((d,d))
    for k in range(r):
        for j in range(len(X[k])):
            term = X[k][j] - X[k].mean(axis=0)
            E += np.outer(term,term)
    
    W = np.zeros((d,d))
    X_bar = dfs.mean(axis=0)
    for k in range(r):
        for j in range(len(X[k])):
            term = X[k][j] - X_bar
            W += np.outer(term,term)
    
    B = W - E 
    # 有时也记为 B,W,T。 按B,E,W记的话
    # B是组间离差阵，E是组内离差阵，W是两者之和
    # 此这可以求出wilks lambda 统计量，即 |E|/|W|
    # 即组间误差占总误差的数量.该统计量又可以经过一些变换转化为
    # F统计量进行F检验，检查其偏离模型没用的假设是否被拒绝
    # 或者也可以直接在wilks分布上搞
    return B,E,W
    
class wilks_lambda(object):
    def __init__(self, p, n1, n2):
        self.p = p
        self.n1 = n1
        self.n2 = n2
        
        t = n1 + n2 - (p+n2+1)/2
        s = np.sqrt((p**2*n2**2-4)/(p**2+n2**2-5))
        lam = (p*n2-2)/4
        
        self.t = t
        self.s = s
        self.lam = lam
        
        n1_f = p*n2
        n2_f = np.round(t*s-2*lam)
        
        self.n1_f = n1_f
        self.n2_f = n2_f
        
        self.f = stats.f(n1_f,n2_f)
    def R(self, Lam):
        t,s,lam,n2,p = self.t,self.s,self.lam,self.n2,self.p
        return (1-Lam**(1/s))/Lam**(1/s) * (t*s-2*lam)/(p*n2)
    def pdf(self,Lam):
        return self.f.pdf(self.R(Lam))
    def cdf(self,Lam):
        return self.f.cdf(self.R(Lam))
    def isf(self,Lam):
        return self.f.isf(self.R(Lam))

w_stat = np.linalg.det(E)/np.linalg.det(W)
w = wilks_lambda(4,32,2)
p_value = 1 - w.cdf(w_stat)
print(p_value)