import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import cPickle as pkl
from sklearn.preprocessing import normalize
#from networkx.generators.random_graphs import fast_gnp_random_graph
#from networkx.convert import to_scipy_sparse_matrix

#def makedata(time,size,prob):
#    graphs = [to_scipy_sparse_matrix(fast_gnp_random_graph(size,prob),dtype=int) for ii in range(time)]
#    return graphs

def readdata(filename, firstday, lastday, directed = False, normalized = False, aggregated = False):
    """ Reading data from a textfile formatted as follows:
        first_node | second_note | timestep.
        Returns a Dictionary (key=timestep value=adjacency-matrix)
    """
         
    # Read the data. We only consider the first three columns, i.e. no weights are included
    source, target, time = np.loadtxt(filename, dtype = int, usecols=(0,1,2), unpack=True)
    if not firstday:
        firstday = min(time)
    if not lastday:
        lastday = max(time)
    
    # Select only edges within the given time constraint
    index = np.where(np.logical_and(time >= firstday, time <= lastday))
    source, target, time = source[index], target[index], time[index]
    
    #if "aggregated" is an integer > 1 then it determines the number of days, which are aggregated. 
    #Otherwise "aggregated" is False by default or True for full aggregation
    if  isinstance(aggregated,int):
        if aggregated > 1:
            old = np.arange(firstday,lastday+1)
            days = lastday-firstday+1
            new = np.repeat(np.arange(int(np.ceil(1.*days/aggregated))),aggregated)[0:days]
            old_to_new = dict(zip(old, new))
            map_old_to_new = np.vectorize(old_to_new.__getitem__)
            time = map_old_to_new(time)
    
    # Reindex nodes old -> 0...N_vertices-1
    nodes = set(source) | set(target)
    number_of_nodes = len(nodes)
    old_to_new = {old:new for new,old in enumerate(nodes)}
    map_old_to_new = np.vectorize(old_to_new.__getitem__)
    edges = zip(map_old_to_new(source), map_old_to_new(target), time)
    
    if isinstance(aggregated,bool) and aggregated is True:
        Static_Network = get_Static_Network(edges,firstday,lastday,directed,number_of_nodes,normalized)
        return Static_Network
    else:
        Temporal_Network = get_Temporal_Network(edges,firstday,lastday,directed,number_of_nodes,normalized)
        return Temporal_Network


def get_Temporal_Network(edges,firstday,lastday,directed,number_of_nodes,normalized):
    # Dictionary indexed by times from 0 to firstday-lastday: time: edge_list
    time_to_edges = {t: set() for t in xrange(0, lastday-firstday+1)}
    for u,v,t in edges:
        if u != v: # ignore self loops
            time_to_edges[t - firstday].add((u,v))
            if not directed:
                time_to_edges[t - firstday].add((v,u))
    # Initialize the temporal network
    Temporal_Network = {}
    for time, edges in time_to_edges.items():
        col = [u for u,v in edges]
        row = [v for u,v in edges]
        dat = [True for i in range(len(edges))]

        Adj_Matrix = sp.csr_matrix((dat,(row,col)),
                shape=(number_of_nodes, number_of_nodes), dtype=bool)
        # !!!!!!!!! Annahme, dass Kante: u -> v und p(t+1) = Ap(t) bzw. A[v,u] = 1 !!!!!!!!
        if normalized:
            Adj_Matrix = normalize(Adj_Matrix.transpose(), norm='l1', axis=1, copy=False).transpose()
            Temporal_Network[time] = Adj_Matrix
        else:
            Temporal_Network[time] = Adj_Matrix
    return Temporal_Network

def get_Static_Network(edges,firstday,lastday,directed,number_of_nodes,normalized):
    edges = set([(u,v) for u,v,t in edges if u != v])
    if not directed:
        edges = edges | set([(v,u) for u,v in edges])
    
    col = [u for u,v in edges]
    row = [v for u,v in edges]
    dat = [True for i in xrange(len(edges))]
    Adj_Matrix = sp.csr_matrix((dat,(row,col)),shape=(number_of_nodes, number_of_nodes), dtype=bool)
    # !!!!!!!!! Annahme, dass Kante: u -> v und p(t+1) = Ap(t) bzw. A[v,u] = 1 !!!!!!!!
    if normalized:
        Adj_Matrix = normalize(Adj_Matrix.transpose(), norm='l1', axis=1, copy=False).transpose()
    
    Static_Network = {t:Adj_Matrix for t in xrange(lastday-firstday+1)}
    return Static_Network

def save_sparse_csr(filename,array):
    time    = len(array)
    data    = {}
    indices = {}
    indptr  = {}
    shape   = {}

    for ii in range(time):
        data[ii]    = array[ii].data
        indices[ii] = array[ii].indices
        indptr[ii]  = array[ii].indptr
        shape[ii]   = array[ii].shape

    newarray = [data, indices, indptr, shape]
    pkl.dump( newarray, open( filename, "wb" ) )

def load_sparse_csr(filename):

    array    = pkl.load( open (filename, "rb") )
    time     = len(array[0])
    newarray = {}

    for ii in range(time):
        data         = array[0][ii]
        indices      = array[1][ii]
        indptr       = array[2][ii]
        shape        = array[3][ii]
        matrix       = sp.csr_matrix( (data,indices,indptr), shape , dtype=bool)
        newarray[ii] = matrix

    return newarray

if __name__ == "__main__": # (Als Testumgebung)
    A = readdata('sociopatterns_hypertext.dat',0,0)
    save_sparse_csr('sociopatterns_hypertext.p',A)
    print A[2232]

