import numpy as np
from medusa.graph_theory import degree,modularity

def participation_coefficient(W):
    """
    Calculates the participation coefficient, which is the diversity of the 
    between-module connections of each node

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    nodal_part : numpy array
        Nodal participation coefficient.

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')

    N = W.shape[0] # Graph size       
        
    _,community = modularity.modularity(W, 1) # Community vector
    
    deg = degree.degree(W, 'CPU') # Degree of each node
    
    Gc = np.matmul(np.where(deg != 0, 1,0)[:,np.newaxis],community.T) # Multiplying each graph row by its community vector
    
    # Add iteratively the participation of each node in each group
    Kc = np.zeros((N,1))
    for ii in range(1,np.amax(community).astype(int)+1):
        Kc = Kc + (np.sum(W * np.where(Gc==ii,1,0),axis=1)**2)[:,np.newaxis]
        
    # Guimera and Amaral normalisation
    nodal_part = np.ones((N,1)) - Kc / (deg.T**2)[:,np.newaxis]
    nodal_part[deg == 0] = 0
    
    return nodal_part

# import scipy.io as rmat

# data = rmat.loadmat('D:/OneDrive - Universidad de Valladolid/Scripts/testPython/graphTest.mat')
# W = data['W']
# aa = participation_coefficient(W)