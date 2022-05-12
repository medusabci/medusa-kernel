import numpy as np


def eigen_centrality(W):
    """
    Calculates the eigenvector centrality, which is a centrality measure based
    on the adjacency matrix eigenvectors
    
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    mode : string
        GPU or CPU
        
    Returns
    -------
    nodal_eig : numpy array
        Nodal eigenvector centrality

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')        
      
    N = W.shape[0] # Number of nodes
    
    D,V = np.linalg.eig(W)
    
    idx = np.where(D == max(D))[0]
    
    nodal_eig = abs(V[:,idx])
    
        
    return nodal_eig

# import scipy.io as rmat

# data = rmat.loadmat('D:/OneDrive - Universidad de Valladolid/Scripts/testPython/graphTest.mat')
# W = data['W']
# W = np.squeeze(W)
# aa = eigen_centrality(W)