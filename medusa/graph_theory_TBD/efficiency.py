import numpy as np

def __distance_inv_wei(L):
    """
    Inverse distance matrix. Dijkstra's algorithm. Rubinov & Sporns 2010

    Parameters
    ----------
    L : numpy 2D matrix
        Connection-length matrix THIS IS NOT WEIGHTS MATRIX

    Returns
    -------
    D : numpy 2D matrix
        Distance matrix.
    B : numpy 2D matrix
        Number of edges matrix. 
    """
    n = L.shape[0]       
    D = np.ones((n,n)) * np.inf # Distance matrix
    np.fill_diagonal(D,0)
    
    for u in range(0,n):
        S = np.ones((1,n),dtype=np.int32) # Distance permanence placeholder
        L1 = np.copy(L)
        V = [[u]]
        while 1:
            S[0,V] = 0 # Distance u>V is now permanent
            L1[:,V] = 0 # Remove in-edges
            for v in V:
                T = np.argwhere(L1[v,:]) # Neighbours of shortest nodes
                D[u,T[:,1]] = np.min(np.concatenate((D[u,T[:,1]][:,np.newaxis], (D[u,v]+L1[v,T[:,1]])[:,np.newaxis]),axis=1),axis=1)
                
            if (D[u,S[0,:].astype(bool)] == 0).all(): # Empty = all nodes reached, Inf = some nodes cannot be reached
                break
            
            minD = np.min(D[u,S[0,:].astype(bool)])

            if minD == np.inf: # Empty = all nodes reached, Inf = some nodes cannot be reached
                break
        
            V = np.argwhere(D[u,:]==minD)

    np.fill_diagonal(D,np.inf)    
    D = 1/D
    
    return D   

def efficiency(W):
    """
    Calculates the graph efficiency. 
    Globally is the mean of the inverse shortests path length, and is 
    inversely related to path length. 
    Nodally/Locally is the same as globally but calculated on the neighbourhood of the
    node, and is related with clustering coefficient. 
    
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    global_eff : numpy array
        Global efficiency.   
    nodal_eff : numpy array
        Nodal/Local efficiency.

    """
    
    N = W.shape[0] # Number of graph nodes
    W1 = np.copy(W) # Graph copy
    A = np.where(W != 0,1,0) # Connections matrix
    idx = np.copy(A)
    W1 = np.where(idx,1/W1,0) # Distance matrix (inverse of the non-zero weights)
    
    # Global efficiency
    e = __distance_inv_wei(W1)
    global_eff = np.sum(np.squeeze(np.asarray(e))) / (N**2 - N)
    
    # Nodal/Local efficiency - Based on Rubinov's BCT toolbox
    nodal_eff = np.zeros((N,1))
    for u in range(0,N):
        V = np.argwhere(np.logical_or(A[u,:],A.T[:,u])) # Neighbours
        sw = W[u,V]**(1/3) + W.T[u,V]**(1/3) # Symmetrized weights vector
        W1_tmp = np.copy(W1[np.squeeze(V),:])
        e = __distance_inv_wei(W1_tmp[:,np.squeeze(V)]) # Inverse distance matrix
        se = e**(1/3) + e.T**(1/3) # Symmetrized inverse distance matrix
        num = np.sum(np.sum((sw.T*sw)*se))/2 # Numerator

        if num != 0:
            sa = A[u,V] + A.T[u,V] # Symmetrized adjacency vector
            den = np.sum(sa)**2 - np.sum(sa**2) # Denominator
            nodal_eff[u] = num / den # Nodal/Local efficiency
            
    return global_eff,np.squeeze(nodal_eff)
