  # -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 01:21:30 2021

@author: fbrav
"""

import numpy as np
from numpy import array, pi, zeros, sqrt, linalg,matmul
from beam import beam_element

xy = array([
    [0,0],      #0
    [6,0],      #1
    [0,3],      #2
    [6,3],      #3
    [0,5],      #4
    [6,5],      #5
    [0,6],      #6
    [6,6.5]])   #7

conec = array([
    #columns
    [0,2],      #0 
    [1,3],      #1
    [2,4],      #2
    [3,5],      #3
    [4,6],      #4
    [5,7],      #5
    #beam
    [2,3],      #6
    [4,5],      #7
    #roof
    [6,7]])      #8

dofs_rest = np.array([0,1,3,4])
rest_val = np.array([0,0,0,0])
dofs_f = np.array([2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])

g= 9.82 #m/s**2
gamma = g*2400 # N / m3
b1 = 0.3 
h1 = 0.3
b2 = 0.2 
h2 = 0.4
b3 = 0.2
h3 = 0.2


properties_column = {}
properties_column["E"] = 4e10
properties_column["A"] = b1*h1
properties_column["I"] = b1*h1**3/12
properties_column["qx"] = -gamma*b1*h1
properties_column["qy"] = 0.0

properties_beam = {}
properties_beam["E"] = 4e10
properties_beam["A"] = b2*h2
properties_beam["I"] = b2*h2**3/12
properties_beam["qx"] = 0.0
properties_beam["qy"] = -gamma*b2*h2

L_roof = np.sqrt(1.5**2+6.0**2)
sintet_roof = 1.5/L_roof
costet_roof = 6.0/L_roof

properties_roof = {}
properties_roof["E"] = 4e10
properties_roof["A"] = b3*h3
properties_roof["I"] = b3*h3**3/12
properties_roof["qx"] = -gamma*b3*h3*sintet_roof
properties_roof["qy"] = -gamma*b3*h3*costet_roof
 

Nnodes = xy.shape[0]
Nelems = conec.shape[0]
Nres = dofs_rest.shape[0]
Ndof = dofs_f.shape[0]
NDOFs_node = 3
NDOFs = NDOFs_node*Nnodes 

K = zeros((NDOFs, NDOFs))
Kff = zeros((NDOFs-Nres, NDOFs-Nres))
Kfc = zeros((NDOFs-Nres, Nres))
Kcf = zeros((Nres, NDOFs-Nres))
Kcc = zeros((Nres, Nres))
f = zeros((NDOFs, 1))
ff = zeros((NDOFs-Nres, 1))
fc = zeros((Nres, 1))
uc = zeros((Nres, 1))


for e in range(Nelems):
    ni = conec[e,0]
    nj = conec[e,1]
    
    print(f"e = {e} ni = {ni} nj = {nj}")
    
    xy_e = xy[[ni, nj], :]
    
    if e in [0,1,2,3,4,5]:
        ke, fe = beam_element(xy_e, properties_column)
    if e in [6,7]:
        ke, fe = beam_element(xy_e, properties_beam)
    if e == 8:
        ke, fe = beam_element(xy_e, properties_roof)
    
    
    d = [3*ni, 3*ni+1, 3*ni+2, 3*nj, 3*nj+1, 3*nj+2] #global DOFs from local DOFs
    
    #direct stiffness method
    for i in range(2*NDOFs_node):
        p = d[i]
        for j in range(2*NDOFs_node):
            q = d[j]
            K[p,q] += ke[i,j]
        f[p]+= fe[i]

countx=0
for x in dofs_f:
    county=0
    ff[countx]=f[x]
    for y in dofs_f:
        Kff[countx,county] = K[x,y]
        county+=1
    countx+=1
#print(Kff)
#print(ff)

countx=0
for x in dofs_rest:
    county=0
    fc[countx]=f[x]
    uc[countx]=rest_val[countx]
    for y in dofs_rest:
        Kcc[countx,county] = K[x,y]
        county+=1
    countx+=1
#print(Kcc)
#print(fc)

countx=0
for x in dofs_f:
    county=0
    for y in dofs_rest:
        Kfc[countx,county] = K[x,y]
        Kcf[county,countx] = K[x,y]
        county+=1
    countx+=1
#print(Kcf)
#print(Kfc)
uf = np.linalg.solve(Kff, ff-np.matmul(Kfc,uc))

print(f"K^-1 = {np.linalg.inv(Kff)}")
print(f"Kff = {Kff}")
print(f"Kff = {Kff}")
print(f"Kfc = {Kfc}")
print(f"Kcf = {Kcf}")
print(f"ff = {ff}")
print(f"fc = {fc}")
print(f"uf = {uf}")
print(f"uc = {uc}")



######------------######


import matplotlib.pyplot as plt
from matplotlib import collections  as mc


fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
# Nodos


plt.scatter(xy[:,0], xy[:,1])

for n in range(xy.shape[0]):
    plt.text(xy[n,0], xy[n,1], f"{n}")

lines = [[(0,0),(0,3)],     #0
         [(6,0),(6,3)],     #1
         [(0,3),(0,5)],     #2
         [(6,3),(6,5)],     #3
         [(0,5),(0,6)],     #4
         [(6,5),(6,6.5)],   #5
         [(0,3),(6,3)],     #6
         [(0,5),(6,5)],     #7
         [(0,6),(6,6.5)]]   #8

deform = [[(0+0,0+0),(0-5.45980621e-04,3-1.76425554e-05)],     #0
         [(6+0,0+0),(6-5.51476745e-04,3-2.26721763e-05)],     #1
         [(0-5.45980621e-04,3-1.76425554e-05),(0-7.17066296e-04,5-2.39515299e-05)],     #2
         [(6-5.45980621e-04,3-1.76425554e-05),(6-7.17284313e-04,5-3.10628896e-05)],     #3
         [(0-7.17066296e-04,5-2.39515299e-05),(0-8.27167083e-04,6-2.49014427e-05)],     #4
         [(6-7.17284313e-04,5-3.10628896e-05),(6-8.13251141e-04,6.5-3.30768361e-05)],   #5
         [(0-5.45980621e-04,3-1.76425554e-05),(6-5.51476745e-04,3-2.26721763e-05)],     #6
         [(0-7.17066296e-04,5--2.39515299e-05),(6-7.17284313e-04,5-3.10628896e-05)],     #7
         [(0-8.27167083e-04,6-2.49014427e-05),(6-8.13251141e-04,6.5-3.30768361e-05)]]   #8

lc = mc.LineCollection(lines, colors='blue', linewidths=1)
lf = mc.LineCollection(deform, colors='red', linewidths=1)

plt.text(-0.3,1.5, f"{[0]}")
plt.text(6.1,1.5, f"{[1]}")
plt.text(-0.3,4, f"{[2]}")
plt.text(6.1,4, f"{[3]}")
plt.text(-0.3,5.5, f"{[4]}")
plt.text(6.1,5.75, f"{[5]}")
plt.text(3,2.6, f"{[6]}")
plt.text(3,4.6, f"{[7]}")
plt.text(3,5.8, f"{[8]}")
ax.add_collection(lc)
ax.add_collection(lf)
