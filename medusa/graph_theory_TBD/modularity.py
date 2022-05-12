import numpy as np

def modularity(W,gamma):
    """
    Calculates the optimal node comunity and the modularity
    
    Note: The optimal community structure subdivides the network in groups of 
    nodes (non-overlapping), maximizing withing-groups edges while minimising 
    between-groups edges. Modularity quantifies the degree to which the graph 
    could be subdivided into the aforementioned communities
        
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    gamma : float
        modularity resolution parameter     
            gamma>1     smaller modules
            0<=gamma<1  larger modules
            gamma=1     (default) classic modularity function
        
    Returns
    -------
    Q : int
        Network modularity. 
    Ci : numpy array
        Network communities.
        
    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')  
    
    if 'gamma' not in locals():
        gamma = 1 # Default
        
    N = W.shape[0]
    n_perm = np.random.permutation(N) - 1
    W = W[:, n_perm][n_perm] # Randomly permute the order of the nodes
    K = np.sum(W,axis=0) # Degree
    K = K[np.newaxis,:]
    m = np.sum(K) # # of edges (undirected edges are counted twice)
    B = W - gamma * (np.matmul(K.T,K)) / m # Modularity matrix
    Ci = np.ones((N,1)) # Community indices
    cn = 1 # Number of communities
    U = np.array([1, 0]) # Array of unexamined communities
    
    ind = np.arange(N)
    ind = ind[:,np.newaxis]
    Bg = np.copy(B)
    Ng = N
    
    while U[0]: # Community U[0]
        D,V = np.linalg.eig(Bg)
        il = np.argmax(np.real(D))
        v1 = V[:,il] # Eigenvector of the max eigenvalue of Bg
        
        S = np.ones((Ng,1))
        S[v1<0,0] = -1
        q = np.matmul(np.matmul(S.T,Bg),S) # Contribution to modularity

        if q > 10**-10: # Contribution positive: U[0] is divisible
            qmax = q # Max contribution to modularity
            Bg[np.eye(Ng,dtype=np.bool)] = 0 # To enable fine-tuning, modify Bg
            indg = np.ones((Ng,1)) # Array of unmoved indices
            Sit = np.copy(S)
            while not (np.isnan(indg)).all(): # Iterative fine-tuning
                Qit = qmax - 4 * Sit * np.matmul(Bg,Sit)
                qmax = np.nanmax(Qit * indg)
                imax = (Qit == qmax)
                Sit[imax[:,0],0] = -Sit[imax[:,0],0]
                indg[imax[:,0],0] = np.nan
                if qmax > q:
                    q = qmax
                    S = np.copy(Sit)
                    
            if abs(np.sum(S)) == Ng: # Unsuccessful splitting of U[0]
                U[0] = []
            else:
                cn = cn + 1
                Ci[ind[S[:,0]==1][:,0],0] = U[0] # Split old U[0] into new U[0] and into cn
                Ci[ind[S[:,0]==-1][:,0],0] = cn
                U = np.insert(U,0,cn)
        else: # Contribution nonpositive: U[0] is indivisible
            U = np.delete(U, 0)
            
        ind = np.argwhere(Ci == U[0]) # Indices of unexamined community U[0]
        bg = B[:,ind[:,0]][ind[:,0]]
        Bg = bg - np.diag(np.sum(bg,axis=0)) # Modularity matrix for U[0]
        Ng = ind.shape[0] # Number of vertices in U[0]

    s = np.tile(Ci, (1, N)) # Compute modularity
    Q = (s - s.T)
    Q[Q!=0] = 1
    Q = np.logical_not(Q).astype(int) * B / m
    Q = np.sum(np.squeeze(np.asarray(Q)))
    Ci_corrected = np.zeros((N,1)) # Initialise Ci_corrected
    Ci_corrected[n_perm,0] = Ci[:,0] # Return order of nodes to that used in the input stage
    Ci = Ci_corrected # Output corrected community assignment

    return Q,Ci
