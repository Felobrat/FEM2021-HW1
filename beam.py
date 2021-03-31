# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:56:12 2021

@author: fbrav
"""


from numpy import array, arctan2, zeros, ix_
import numpy as np
from scipy.linalg import norm

def beam_element(xy, properties):
    E = properties["E"]
    I = properties["I"]
    A = properties["A"]
    
    qx = properties["qx"]
    qy = properties["qy"]
    
    xi = xy[0, :]
    xj = xy[1, :]
    
    L = norm(xj-xi)
    #\theta = arctan2(xj[1]-xi[1], xj[0] - xi[0])
    
    costheta = (xj[0] - xi[0])/L
    sintheta = (xj[1] - xi[1])/L
    
    ke = np.zeros((6,6))
    fe = np.zeros((6,1))
    fe_tilde = np.zeros((6,1))
    ke_tilde = np.zeros((6,6))
    
    ke_tilde[0,0] = A*E/L
    ke_tilde[3,3] = A*E/L
    ke_tilde[0,3] = -A*E/L
    ke_tilde[3,0] = -A*E/L
    
    fe_tilde[0,0] = qx*L/2
    fe_tilde[1,0] = qy*L/2
    fe_tilde[2,0] = qy*L**2/12
    fe_tilde[3,0] = qx*L/2
    fe_tilde[4,0] = qy*L/2
    fe_tilde[5,0] = -qy*L**2/12
    
    bending_dofs = np.ix_([1,2,4,5],[1,2,4,5])
    
    ke_tilde[bending_dofs] = E*I*np.array(
        [[ 12/L**3     ,  6/L**2  , -12/L**3  ,   6/L**2 ],
         [  6/L**2     ,  4/L     ,  -6/L**2  ,   2/L    ],
         [-12/L**3     , -6/L**2  ,  12/L**3  ,  -6/L**2 ],
         [  6/L**2     ,  2/L     ,  -6/L**2  ,   4/L    ]])
    
    T = np.zeros((6,6))
    
    T[0,0] = costheta
    T[1,1] = costheta
    T[2,2] = 1.0
    T[3,3] = costheta
    T[4,4] = costheta
    T[5,5] = 1.0
    T[0,1] = sintheta
    T[1,0] = sintheta
    T[3,4] = sintheta
    T[4,3] = sintheta
    
    ke = T@ke_tilde@T.T
    fe = T@fe_tilde
    return ke, fe
    
    