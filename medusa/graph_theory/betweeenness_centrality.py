import numpy as np

def betweenness(W,norm = True):
    """
    Calculates the betweenness centrality of the graph, which is the probability 
    that node i belong to one of the network shortest paths
    
    Note: W must be converted to a connection-length matrix. It is common to 
    obtain it via mapping from weight to length. BC is normalised to the 
    range [0 1] as BC/[(N-1)(N-2)]
        
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    norm : bool
        Normalisation. 0 = no, 1 = yes. Default = True

    Returns
    -------
    BC : numpy array
        Network betweenness centrality.
        
    """
    if not (norm == 0 or norm == 1 or norm == True or norm == False):
        raise ValueError('Variable "norm" must be either 0, 1, or bool')
    # norm = norm.astype(bool)        
        
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')      
    
    n = W.shape[0]
    E = np.argwhere(W)
    W[E] = 1/W[E] # Invert weights
    BC = np.zeros((n,1)) # Vertex betweenness
    
    for u in range(0,n):
        D = np.ones((1,n)) * np.inf
        D[0,u] = 0 # Distance from u
        NP = np.zeros((1,n))
        NP[0,u] = 1 # Number of paths from u
        S = np.ones((1,n)) # Distance permanence (1 is not permanent)
        P = np.zeros((n,n)) # Predecessors
        Q = np.zeros((1,n))
        q = n - 1# Order of non-increasing distance
        
        W1 = np.copy(W)
        V = [u]
        
        while 1:
            S[0,V] = 0 # Distance u>V is now permanent
            W1[:,V] = 0 # No in-edges as already shortest
            for v in V:
                Q[0,q] = v + 1
                q = q - 1
                G = np.argwhere(W1[v,:]) # Neighbours of v
                for g in G:
                    Duw = D[:,v] + W1[v,g] # Path length to be tested
                    
                    if Duw < D[0,g]: # If new u>W shorter than old
                        D[0,g] = Duw
                        NP[0,g] = NP[0,v] # NP(u>w) = NP of new path
                        P[g,:] = 0
                        P[g,v] = 1 # v is the only predecessor
                    elif Duw == D[0,g]: # if new u>v equal to old
                        NP[0,g] = NP[0,g] + NP[0,v] # NP(u>w) sum of old and new
                        P[g,v] = 1 # v is also a predecessor
             
            if not D[:,np.squeeze(S).astype(bool)].any(): # All nodes reached...
                break
            
            minD = np.min(D[:,np.squeeze(S).astype(bool)])

            if minD == np.inf: # ...some cannot be reached:
                Q[:,0:q+1] = np.transpose(np.argwhere(np.squeeze(D) == np.inf)) # ...these are first-in-line
                break
            
            V = np.argwhere(D==minD)
            V = V[:,1]
            
        DP = np.zeros((n,1)) # Dependency
        for w in (Q[0,0:n-1]-1).astype(int):
            BC[w,0] = BC[w,0] + DP[w,0]
            for v in np.argwhere(P[w,:]):
                DP[v,0] = DP[v,0] + (1 + DP[w,0]) * NP[0,v] / NP[0,w]
    if norm:
        BC = BC / ((n - 1) * (n - 2))
                
    return BC            
        