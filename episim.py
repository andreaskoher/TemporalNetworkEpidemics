#!/usr/bin/python2.7
from numpy import savez
from readfile import readdata
import epimod

class EpiSim():
    """
    The constructor expects the filename of a comma seperated edgelist (source,
    target, time, weight). It constructs a dict with time steps as keys 
    a sparse matrix (default: csr) for each corresponding values. The result is
    stored in self.A (adjacency matrix).
    
    Parameters
    ----------
    fname : str
        File name of the edge list. Format: source, target, time, weight.
    directed : bool
        Interpret edge list entry as undirected links (default)
    observation_window: (int, int)
        Choose the first and last time step for the simulation. By default, the
        full edgelist will be evaluated.
    aggregated : bool
        Ignore time information to get a fully aggregated network. Default: False
    normalized : bool
        Normalize the out-going edge weights for each node. Default: False  
    """
    def __init__(self,
                 edgelist_fname        = "sexual_contacts.dat",
                 directed              = False,
                 observation_window    = (None,None),
                 aggregated            = False,
                 normalized            = False):
        
        
        firsttime = 0 if observation_window[0] is None else observation_window[0]
        lasttime = 0 if observation_window[1] is None else observation_window[1]
        if isinstance(edgelist_fname, basestring):
            self.A = readdata(edgelist_fname, int(firsttime), int(lasttime),
                              directed=bool(int(directed)), normalized=bool(int(normalized)),
                              aggregated=bool(int(aggregated)))
        
        self.nodes   = self.A[0].shape[0]
        self.runtime = len(self.A)
        self.edgelist_fname    = str(edgelist_fname)
        self.directed          = bool(int(directed))
        self.start             = int(firsttime)
        self.end               = int(lasttime) if int(lasttime) > 0 else len(self.A)
        self.normalized        = bool(int(normalized))
        self.aggregated        = bool(int(aggregated))
        self.initialize()
        
        print "=================== Initialized the Temporal Network ==================="
        print "nodes:            ", self.nodes
        print "snapshots        : ", self.runtime
        print "observation start:", self.start
        print "observation end  :", self.end
        print "normalized       :", self.normalized
        print "aggregated       :", self.aggregated
    
    def initialize(self):
        self.mean_cumulative   = None
        self.mean_prevalence   = None
        self.mean_incidence    = None
        self.single_cumulative = None
        self.single_prevalence = None
        self.single_incidence  = None
        self.infection_arrival_time = {}
        self.accessibility_matrix = {"data": None, "indices": None, "indptr": None}
        self.recovery          = None
        self.infection_arrival_time = None
        
    def save(self, fname = "simulation"):
        print "save in ", fname, ".npz"
        savez(fname,
              edgelist_fname = self.edgelist_fname,
              directed = self.directed,
              nodes=self.nodes,
              runtime=self.runtime,
              mean_cumulative=self.mean_cumulative,
              mean_prevalence=self.mean_prevalence,
              mean_incidence=self.mean_incidence,
              single_cumulative=self.single_cumulative,
              single_prevalence=self.single_prevalence,
              single_incidence=self.single_incidence,
              accessibility_matrix = self.accessibility_matrix,
              recovery=self.recovery,
              start=self.start,
              end=self.end,
              infection_arrival_time=self.infection_arrival_time,
              normalized=self.normalized,
              aggregated=self.aggregated)
    
    def run_disease(self,
                    model = "SIR",
                    recovery = 14,
                    memory_efficient = True,
                    track_single_outbreak = False,
                    save_matrix = False,
                    verbose = True,
                    simulation_time=0,
                    infection_arrival_time=False,
                    alpha = 1.):
        """
        Start the simulation.
        
        Parameters
        ----------
        model : str
            choose an epidemiological model. Default: 'SIR'
        recovery : int
            number of time steps to recovery (ignored for model = "SI")
        memory_efficient : bool
            compute the full reachability matrix at once or go through each
            seeding node individually to save RAM (much slower though).
            Default: False
        track_single_outbreak : bool
            save the incidence and prevalence levels for all nodes individually.
            Default: False
        save_matrix : bool
            save reachability matrix. Default: False
        verbose : bool
            print output. Default: True
        simulation_time : int
        alpha : float
            choose an infection transmission probability. For alpha < 1 the 
            simulations have to be repeated to get an average statistics.
        """
        self.recovery = int(recovery)
        
        if simulation_time == 0:
                simulation_time = len(self.A)
        if verbose:
            print "simulation time:", simulation_time,"\n\n"
        
        if model == 'SIR':
            epimod.sir(self,
                       memory_efficient,
                       track_single_outbreak,
                       save_matrix,
                       verbose,
                       simulation_time,
                       infection_arrival_time,
                       alpha)
        
        if model == 'SI':
            epimod.si(self,
                       memory_efficient,
                       track_single_outbreak,
                       save_matrix,
                       verbose,
                       simulation_time,
                       infection_arrival_time,
                       alpha)
        if model == 'SIS':
            epimod.sis(self,
                       memory_efficient,
                       track_single_outbreak,
                       save_matrix,
                       verbose,
                       simulation_time,
                       infection_arrival_time,
                       alpha)
        else:
            pass

#===============================================================================
# Es folgt die Testumgebung:
#===============================================================================
#load either sociopatterns_hypertext.dat, pig_trade_11-14_uvdw_from_v0_2.dat,  sexual_contacts.dat or the test network
if __name__ == "__main__":
    import scipy.sparse as sp
    import numpy as np
    #A = []
    #A.append(sp.csr_matrix(([1,1], [[2,1],[1,2]]),shape=(4,4)))
    #A.append(sp.csr_matrix(([1,1], [[3,2],[2,3]]),shape=(4,4)))
    #A.append(sp.csr_matrix(([1,1], [[3,0],[0,3]]),shape=(4,4)))
    A = EpiSim(edgelist_fname = "/home/andreas/reachability/sociopatterns_hypertext.dat",
               directed = 0,
               observation_window=(0,2999),
               aggregated=0)
    A.run_disease(model                 = "SIS",
                  recovery              = 1000, 
                  memory_efficient      = False, 
                  track_single_outbreak = True, 
                  save_matrix           = False, 
                  verbose               = False, 
                  simulation_time       = 0,
                  alpha = 1)