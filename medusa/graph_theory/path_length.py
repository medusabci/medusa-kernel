import numpy as np

def __charpath(D,diagonal_dist=None,infinite_dist=None):
    """
    Path length, global efficiency and others
    
    Note: The input distance matrix require distance_wei. Path length is the 
    mean of all the nodal path lengths. Infinite paths (disconected nodes) are
    included by default. The parameter infinite_dist may madify this

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    diagonal_dist : numpy array
        Distances on the main diagonal. Optional. Default: diagonal_dist = 0        
    infinite_dist : numpy array
        Include infinite distances in calculations. Optional. Default: infinite_dist = 1 
        
    Returns
    -------
    global_pl : int
        Path lenght.
    efficiency : int
        Network efficiency. 
    nodal_ecc : numpy array
        Network eccentricity. 
    radius : int
        Network radius. 
    diameter : int
        Network diameter. 
        
    """
    if np.any(D == np.nan):
        print('Error: distance matrix contains nan values')
    if not diagonal_dist or diagonal_dist  == None:
        np.fill_diagonal(D,np.nan) # Fill diagonal with nans        
    if not infinite_dist and not infinite_dist == None:
        D[D==np.inf] = np.nan # Ignore infinite paths 
        
    Dv = D[np.logical_not(np.isnan(D))] # Get non-nan values
        
    # Mean of the rows of D
    global_pl = np.mean(Dv)
    
    # Efficiency: Mean of the inverse entries of D
    efficiency = np.mean(1./Dv)

    # Eccentricity of each edge
    nodal_ecc = np.nanmax(D,axis=1)

    # Graph radius
    radius = np.min(nodal_ecc)
    
    # Graph diameter
    diameter = np.max(nodal_ecc)
    
    return global_pl, efficiency, nodal_ecc, radius, diameter


def __distance_wei(L):
    """
    Distance matrix (Dijkstra's algorithm'). Contains the length of the 
    shortest path between each pair of nodes.
    
    Notes: lengths between disconected nodes are set to inf. Lengths on the 
    main diagonal are set to 0

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
    B = np.zeros((n,n)) # Number of edges matrix
    
    for u in range(0,n):
        S = np.ones((1,n),dtype=np.int32) # Distance permanence placeholder
        L1 = np.copy(L)
        V = [u]
        while 1:
            S[0,V] = 0 # Distance u>V is now permanent
            L1[:,V] = 0 # Remove in-edges
            for v in V:
                T = np.argwhere(L1[np.squeeze(v),:]) # Neighbours of shortest nodes
                d = np.min(np.concatenate((D[u,T[:,0]][:,np.newaxis], (D[u,v]+L1[v,T[:,0]])[:,np.newaxis]),axis=1),axis=1)
                wi = np.argmin(np.concatenate((D[u,T[:,0]][:,np.newaxis], (D[u,v]+L1[v,T[:,0]])[:,np.newaxis]),axis=1),axis=1)
                D[u,T[:,0]] = d # Smallest of old/new path lengths
                ind = T[wi == 1] # Indices of the lengthed paths
                B[u,ind] = B[u,v]+1 # Increment number of edges in lengthed paths
                
            if (D[u,S[0,:].astype(bool)] == 0).all(): # Empty = all nodes reached, Inf = some nodes cannot be reached
                break
            minD = np.min(D[u,S[0,:].astype(bool)])

            if minD == np.inf: # Empty = all nodes reached, Inf = some nodes cannot be reached
                break
        
            V = np.argwhere(D[u,:]==minD)
        
    return D,B


def path_length(W,diagonal_dist=None,infinite_dist=None):
    """
    Calculates the path length and other graph integration parameters
    
    Note: L must be a connection-length matrix. One way of generating it is
    the inverse of the weight matrix.

    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
        
    Returns
    -------
    global_pl : int
        Path lenght.
    efficiency : int
        Network efficiency. 
    nodal_ecc : numpy array
        Network eccentricity. 
    radius : int
        Network radius. 
    diameter : int
        Network diameter. 
    nodal_d : numpy 2D matrix
        Shortest distances matrix. 
        
    """
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
    
    W = np.array(W)
    if not np.issubdtype(W.dtype, np.number):
        raise ValueError('W matrix contains non-numeric values')
        
    if not diagonal_dist == None and not (diagonal_dist == 0 or diagonal_dist == 1 
            or diagonal_dist == True or diagonal_dist == False):
        raise ValueError('Variable "diagonal_dist" must be either 0, 1, or bool')
        
    if not infinite_dist == None and not (infinite_dist == 0 or infinite_dist == 1 
            or infinite_dist == True or infinite_dist == False):
        raise ValueError('Variable "diagonal_dist" must be either 0, 1, or bool')          
    
    if not infinite_dist == None: infinite_dist = infinite_dist.astype(bool)
    if not diagonal_dist == None: diagonal_dist = diagonal_dist.astype(bool)
    
    L = np.divide(1,W)
    L[L==np.inf] = 0    
    nodal_d,_ = __distance_wei(L)    
    global_pl,efficiency,nodal_ecc,radius,diameter = __charpath(np.copy(nodal_d))
    
    return global_pl,efficiency,nodal_ecc,radius,diameter,nodal_d
