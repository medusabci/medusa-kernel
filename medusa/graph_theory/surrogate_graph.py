import numpy as np

def surrogate_graph(W):
    """
    Calculates the graph degree.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    surrog_W : numpy array
        Surrogated graph.      

    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')        
      
    surrog_matrix = np.zeros((W.shape[0],W.shape[1]))
    idx_up = np.argwhere(np.triu(W, k=1))
    val_up = W[idx_up[:,0],idx_up[:,1]]
    val_up_surrog = val_up[np.random.permutation(val_up.shape[0])]
    surrog_matrix[idx_up[:,0],idx_up[:,1]] = val_up_surrog
    surrog_matrix = surrog_matrix + np.transpose(surrog_matrix)

    return surrog_matrix