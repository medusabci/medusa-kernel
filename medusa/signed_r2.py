# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:33:30 2019

@author: VICTOR
"""
import numpy as np


def r2(class1, class2, signed=True, dim=0):
    """ This function computes the basic form of the squared point biserial
    correlation coefficient (r2-value).
    
        :param class1:  Data that belongs to the first class
        :param class2:  Data that belongs to the second class
        :param signed:  (Optional, default=True) Boolean that controls if the 
        sign should be mantained.
        :param dim:     (Optional, default=0) Dimension along which the r2-value
        is computed. Therefore, if class1 and class2 has dimensions of [observations 
        x samples] and dim=0, the r2-value will have dimensions [1 x samples].
        
        :return r2:     (signed) r2-value.
    """
    # Length of each class
    N1 = class1.shape[dim]
    N2 = class2.shape[dim]
    
    # Pre-computation
    all_data = np.concatenate((class1,class2), axis=dim)    
    v = np.var(all_data, axis=dim)
    m_diff = np.mean(class1, axis=dim) - np.mean(class2, axis=dim) 
    
    # Compute the sign if required
    sign = 1
    if signed:
        sign = np.sign(m_diff)
        sign[sign == 0] = 1
    
    # Final r2 value
    r2 = sign*N1*N2*np.power(m_diff, 2)/(v*np.power(N1+N2, 2))
    return r2
