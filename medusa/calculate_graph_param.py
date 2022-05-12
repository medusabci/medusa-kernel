import numpy as np
from medusa.graph_theory import clustering_coefficient,path_length
from medusa.graph_theory import complexity,density,degree,transitivity
from medusa.graph_theory import betweeenness_centrality,eigen_centrality
from medusa.graph_theory import participation_coefficient,assortativity
from medusa.graph_theory import efficiency,surrogate_graph,modularity

def calculate_graph_param(W,mode,param,surrog = False,nsurrog = 100, norm = True):
    """
    Calculates different graph parameters based on the graph 'W'
        
    Parameters
    ----------
    W : numpy 2D matrix
        Graph matrix. ChannelsXChannels.
    mode : string
        Where to process the data. 'CPU' o 'GPU'
    param : string
        Graph parameter to be calculated
    surrog : bool
        Wheter surrogate the data or not. 0 = no, 1 = yes. Default = True
    nsurrog : bool
        Number of surrogates. Default = 100. Not recomended less than 50        
    norm : bool
        Betweenness centrality normalisation. 0 = no, 1 = yes. Default = True

    Returns
    -------
    global_value : int
        Graph parameter calculated globally.
    nodal_value : numpy array
        Graph parameter calculated nodally/locally.
    config : dictionary
        Configuration parameters. Contains:
            - param: calculated param (string)
            - surrog: surrogated or not (bool)
            - nsurrog: number of surrogates (int)
            - norm: only when param = betweenness centrality. normalize
            results or not (bool)
        
    """
    # Error check    
    if W.shape[0] is not W.shape[1]:
        raise ValueError('W matrix must be square')
        
    if not (surrog == 0 or surrog == 1 
            or surrog == True or surrog == False):
        raise ValueError('Variable "surrog" must be either 0, 1, or bool')  
    
    nsurrog = np.array(nsurrog)                
    if surrog==True and not np.issubdtype(nsurrog.dtype, np.number):
        raise ValueError('Variable "nsurrog" must be numeric')
             
    param = param.lower() # To avoid failures due to capital letters
    mode = mode.upper() # To avoid failures due to capital letters     
    
    # Var initialization
    nodal_value = None
    global_value = None
    
    
    
    ############################ BASIC MEASURES ##############################
    if param == 'degree' or param == 'g':
        nodal_value = degree.degree(W,mode)
        # global_value = np.mean(nodal_value) # Better use density

    elif param == 'density' or param == 'den' or param == 'd':
        global_value,nodal_value = density.density(W,mode)

    elif param == 'eccentricity' or param == 'ecc':
        _,_,nodal_value,_,_,_ = path_length.path_length(W)
        # global_value = np.mean(nodal_value) # Better use radius or diameter        

    elif param == 'radius' or param == 'r':
        _,_,_,global_value,_,_ = path_length.path_length(W)
        if surrog == True:
            global_value_surrog = np.ones((1,nsurrog))
            for s in range(0,nsurrog):
                S = surrogate_graph.surrogate_graph(W)
                _,_,_,global_value_surrog[:,s],_,_ = path_length.path_length(S)
            global_value = (global_value/np.nanmean(global_value_surrog))

    elif param == 'diameter' or param == 'diam':
        _,_,_,_,global_value,_ = path_length.path_length(W)
        if surrog == True:
            global_value_surrog = np.ones((1,nsurrog))
            for s in range(0,nsurrog):
                S = surrogate_graph.surrogate_graph(W)
                _,_,_,_,global_value_surrog[:,s],_ = path_length.path_length(S)
            global_value = (global_value/np.nanmean(global_value_surrog))
                           
            
    ############################## SEGREGATION ###############################
    elif param == 'clustering coefficient' or param == 'clusteringcoefficient' or param == 'clc':
        if surrog == True:
            nodal_value = clustering_coefficient.clustering_coefficient(W, mode)
            nodal_value_surrog = np.ones((W.shape[0],nsurrog))
            for s in range(0,nsurrog):
                S = surrogate_graph.surrogate_graph(W)
                nodal_value_surrog[:,s] = clustering_coefficient.clustering_coefficient(S,mode)
            nodal_value_tmp = nodal_value/np.nanmean(nodal_value_surrog,axis=1)
            global_value = np.mean(nodal_value_tmp)            
        else:
            global_den,_ = density.density(W,mode)
            nodal_value_tmp = clustering_coefficient.clustering_coefficient(W/global_den, mode)
            global_value = np.mean(nodal_value_tmp)
            nodal_value = clustering_coefficient.clustering_coefficient(W, mode)
            
            
    elif param == 'transitivity' or param == 'trans' or param == 'tran' or param == 't':
        global_value = transitivity.transitivity(W,mode)            
        if surrog == True:
            global_value_surrog = np.ones((1,nsurrog))
            for s in range(0,nsurrog):
                S = surrogate_graph.surrogate_graph(W)
                global_value_surrog[:,s] = transitivity.transitivity(S,mode)
            global_value = (global_value/np.nanmean(global_value_surrog))
            
    elif param == 'modularity' or param == 'mod' or param == 'm' or param == 'community': # Not sure if surrogating makes sense here...
        # Modularity may vary (VERY SLIGHTLY) between iterations. Community 
        # may vary more significantly
        global_value,nodal_value = modularity.modularity(W,1) # Modularity = Global; Community = Nodal  

          
    ############################## INTEGRATION ###############################
    elif param == 'path length' or param == 'pathlength' or param == 'pl':
        global_value,_,_,_,_,D = path_length.path_length(W)
        if surrog == True:
            global_value_surrog = np.ones((1,nsurrog))
            for s in range(0,nsurrog):
                S = surrogate_graph.surrogate_graph(W)
                global_value_surrog[:,s],_,_,_,_,_ = path_length.path_length(S)
            global_value = global_value / np.nanmean(global_value_surrog)
            nodal_value = np.mean(D,axis=1)
        else:
            global_den,_ = density.density(W,'CPU')
            global_value,_,_,_,_,_ = path_length.path_length(W/global_den)
            _,_,_,_,_,D = path_length.path_length(W)                 
            nodal_value = np.mean(D,axis=1)
            
    elif param == 'efficiency' or param == 'eff':
        global_value,nodal_value = efficiency.efficiency(W)
        if surrog == True:
            global_value_surrog = np.ones((1,nsurrog))
            for s in range(0,nsurrog):
                S = surrogate_graph.surrogate_graph(W)
                global_value_surrog[:,s],_ = efficiency.efficiency(S)
            global_value = (global_value/np.nanmean(global_value_surrog))                           
            
    ################### SEGREGATION/INTEGRATION BALANCE ######################
    elif (param == 'small-world index' or param == 'small-worldindex'
          or param == 'smallworldindex' or param == 's-wi' or param == 'swi' 
          or param == 'sw'):
        if surrog == True:
            global_value_pl,_,_,_,_,_ = path_length.path_length(W)
            global_value_clc = clustering_coefficient.clustering_coefficient(W,mode)
            global_value_clc = np.mean(global_value_clc)
            global_value = global_value_clc / global_value_pl            
            global_value_pl_surrog = np.ones((1,nsurrog))
            global_value_clc_surrog = np.ones((1,nsurrog))            
            for s in range(0,nsurrog):
                S = surrogate_graph.surrogate_graph(W)
                global_value_pl_surrog[:,s],_,_,_,_,_ = path_length.path_length(S)
                S = surrogate_graph.surrogate_graph(W)
                aux = clustering_coefficient.clustering_coefficient(S,mode)
                global_value_clc_surrog[:,s] = np.mean(aux)
            global_value_pl = global_value_pl / np.nanmean(global_value_pl_surrog)
            global_value_clc = global_value_clc / np.nanmean(global_value_clc_surrog)
            global_value = global_value_clc / global_value_pl
        else:
            global_den,_ = density.density(W,mode)            
            global_value_pl,_,_,_,_,_ = path_length.path_length(W/global_den)
            global_value_clc = clustering_coefficient.clustering_coefficient(W/global_den,mode)
            global_value_clc = np.mean(global_value_clc)
            global_value = global_value_clc / global_value_pl        


    ############################## CENTRALITY ################################
    elif param == 'closeness centrality' or param == 'closenesscentrality' or param == 'cc':
        _,_,_,_,_,D = path_length.path_length(W)
        nodal_value = 1 / np.nanmean(D,axis=1)
        # global_value = np.mean(nodal_value) # Better use Path Length        
        
    elif param == 'betweenness centrality' or param == 'betweennesscentrality' or param == 'bc':
        nodal_value = betweeenness_centrality.betweenness(W,norm)
        # global_value = np.mean(nodal_value) # Better use Path Length        
        
    elif (param == 'eigenvector centrality' or param == 'eigenvectorcentrality'
          or param == 'ec'):
        nodal_value = eigen_centrality.eigen_centrality(W)
        # global_value = np.mean(nodal_value) # Better use Path Length        
        
    elif (param == 'participation coefficient' or param == 'participationcoefficient'
          or param == 'participation' or param == 'part'):
        nodal_value = participation_coefficient.participation_coefficient(W)
        # global_value = np.mean(nodal_value) # Better use Path Length        


    ############################# RESILIENCE #################################
    elif param == 'assortativity' or param == 'assort' or param == 'as':
        global_value = assortativity.assortativity(W,mode)        
        
        
    ##################### REGULARITY AND COMPLEXITY ##########################
    elif (param == 'shannon entropy' or param == 'shannonentropy' 
          or param == 'entropy' or param == 'se' or param == 'h'):
        if mode == 'CPU':
            global_value = complexity.__graph_entropy_cpu(W)
        elif mode == 'GPU':
            global_value = complexity.__graph_entropy_gpu(W)

    elif (param == 'divergency' or param == 'diver' or param == 'dive'):
        if mode == 'CPU':
            global_value = complexity.__divergency_cpu(W)
        elif mode == 'GPU':
            global_value = complexity.__divergency_gpu(W)                        
        
    elif param == 'complexity' or param == 'comp':
        global_value = complexity.complexity(W, mode)
        
    
    else:
        raise ValueError('Unknown parameter')
      
          
    config = {
        'Surrogate (T/F)': surrog,
        'Number of surrogations': nsurrog,
        'Parameter': param        
        }
    
    if (param == 'betweenness centrality' or param == 'betweennesscentrality' 
          or param == 'bc'):
        config.update({'Betweenness centrality normalization': norm})        


    return global_value, nodal_value, config