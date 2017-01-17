from scipy.sparse import csr_matrix,eye
from collections import deque
from numpy import zeros, gradient
from numpy.random import random


def sis(EpiSim, memory_efficient, track_single_outbreak, save_matrix, verbose, simulation_time, infection_arrival_time, transmission_probability):
    """
    =======================================================================
    Susceptible-Infected-Susceptible Model: An infected node transmits the 
    disease with a probability alpha to a neighbor and returns to the 
    susceptible state after a fixed time.
    =======================================================================
    """
    A                         = EpiSim.A
    nodes                     = A[0].shape[0]
    recovery                  = EpiSim.recovery
    memory_efficient          = bool(int(memory_efficient))
    track_single_outbreak     = bool(int(track_single_outbreak))
    save_matrix               = bool(int(save_matrix))
    verbose                   = bool(int(verbose))
    infection_arrival_time    = bool(int(infection_arrival_time))
    get_infection_time        = False
    alpha                     = float(transmission_probability)
    
    if infection_arrival_time:
        EpiSim.infection_arrival_time = {}
        infection_arrival_time = eye(nodes, nodes, 0, format='csr', dtype=float) # arrival times of source nodes = 1
        get_infection_time = True
    if verbose:
        print "=================== Initializing the simulation ==================="
        print "nodes:            ", nodes
        print "simulation time:  ", simulation_time
        print "model:            ", "sis"
        print "recovery time:    ", recovery
        print "memory efficient: ", memory_efficient
        print "track single:     ", track_single_outbreak
        print "save matrix       ", save_matrix
        print "verbose:          ", verbose
        print "infection arrival ", infection_arrival_time
    
    EpiSim.mean_cumulative = zeros((simulation_time,)) #for each time step we save the density of the 
    EpiSim.mean_prevalence = zeros((simulation_time,)) #reachability matrices, which corresponds to the mean
    EpiSim.mean_incidence  = zeros((simulation_time,)) # cumulative, prevalence and incidence
    if track_single_outbreak:
        EpiSim.single_cumulative = zeros((nodes,simulation_time)) #for each time step we save the density of the 
        EpiSim.single_prevalence = zeros((nodes,simulation_time)) #reachability matrices, which corresponds to the mean
        EpiSim.single_incidence  = zeros((nodes,simulation_time))
    
    if not memory_efficient:
        incidence_list = deque(maxlen=recovery) #records matrices, which are relevant for the infection process (first in, last out)
        for ii in xrange(recovery): # initialize the list with empty matrices...
            incidence_list.append(csr_matrix((nodes,nodes), dtype=bool))
        incidence_list.append(eye(nodes,nodes,format='csr', dtype=bool)) # ... and one unit matrix - our initial condition
        cumulative  = eye(nodes,nodes,format='csr', dtype=bool) #we keep a record of the cumulative incidence
        prevalence  = eye(nodes,nodes,format='csr', dtype=bool) #and the prevalence
        print "========================= Start the simulation ================"
        for jj in xrange(0,simulation_time):
            try:
                if verbose: #print progress only if the simulation runs on the local computer
                    if not jj % int(simulation_time/100.):
                        print 'progress: ', round(1.*jj/simulation_time*100) , ' percent , prevalence: ', prevalence.nnz
                contact_matrix = A[jj].copy()
                if not alpha == 1: # remove links if no infection is transmitted
                    contact_matrix.data *= random(contact_matrix.data.shape) < alpha
                    contact_matrix.eliminate_zeros()
                # ============= This is important ============================
                incidence   = contact_matrix.dot(prevalence) # infection transmission
                incidence  -= incidence.multiply(prevalence) # remove previously infected nodes
                prevalence -= incidence_list[0] # remove recovered nodes
                prevalence += incidence # add newly infected nodes to prevalence
                cumulative += incidence # and to the cumulative matrix
                incidence_list.append(incidence) # newly infected nodes are added and recovered nodes are removed implicitly
                # ============================================================
                if get_infection_time:
                    infection_arrival_time = update_infection_arrival_times(incidence, infection_arrival_time, jj)
                if track_single_outbreak:
                    EpiSim.single_cumulative[:,jj] = cumulative.getnnz(axis=0) # save the absolute number of infected
                    EpiSim.single_prevalence[:,jj] = prevalence.getnnz(axis=0) # nodes at time step jj 
                    EpiSim.single_incidence[:,jj]  = incidence.getnnz(axis=0)
                EpiSim.mean_cumulative[jj] = cumulative.nnz # save the absolute number of infected
                EpiSim.mean_prevalence[jj] = prevalence.nnz # nodes at time step jj 
                EpiSim.mean_incidence[jj]  = incidence.nnz
                
            except (KeyboardInterrupt, SystemExit):
                print "Interrupted at: ", jj 
                print "The data will be stored in the file simulation.npz"
                EpiSim.save(fname="simulation")
                break
        EpiSim.mean_cumulative /= float(nodes**2)  # calculate the relative value
        EpiSim.mean_prevalence /= float(nodes**2)
        EpiSim.mean_incidence  /= float(nodes**2)
        if track_single_outbreak:
            EpiSim.single_cumulative /= float(nodes)  # calculate the relative value
            EpiSim.single_prevalence /= float(nodes)
            EpiSim.single_incidence  /= float(nodes)
        
        if get_infection_time:
            EpiSim.infection_arrival_time['data'] = infection_arrival_time.data
            EpiSim.infection_arrival_time['indices'] = infection_arrival_time.indices
            EpiSim.infection_arrival_time['indptr'] = infection_arrival_time.indptr
        
        if save_matrix:
            EpiSim.data = cumulative.data
            EpiSim.indices = cumulative.indices
            EpiSim.indptr = cumulative.indptr
        
    else: # memory efficient (but slow) algorithm
        print "========================= Start the simulation ================"
        for ii in xrange(nodes):
            try:
                if verbose:
                    print 'node ', ii, ' of ', nodes
                
                seed = zeros((nodes,1), dtype=bool)
                seed[ii] = 1. # go through all nodes and take them as the seed of infection
                incidence_list = deque(maxlen=recovery)
                for kk in xrange(recovery):
                    incidence_list.append(csr_matrix((nodes,1), dtype=bool))
                incidence_list.append(csr_matrix(seed, dtype=bool))
                cumulative         = csr_matrix(seed, dtype=bool)
                prevalence         = csr_matrix(seed, dtype=bool)
                
                for jj in range(0,simulation_time):
                    contact_matrix = A[jj].copy()
                    if not alpha == 1: # remove links if no infection is transmitted
                        contact_matrix.data *= random(contact_matrix.data.shape) < alpha
                        contact_matrix.eliminate_zeros()
                    # ============= This is important ============================
                    incidence   = contact_matrix.dot(prevalence) # infection transmission
                    incidence  -= incidence.multiply(cumulative) # remove previously infected nodes
                    prevalence -= incidence_list[0] # remove recovered nodes
                    prevalence += incidence # add newly infected nodes to prevalence
                    cumulative += incidence # and to the cumulative matrix
                    incidence_list.append(incidence) # newly infected nodes are added and recovered nodes are removed implicitly
                    # ============================================================
                    EpiSim.mean_cumulative[jj] += cumulative.nnz
                    EpiSim.mean_prevalence[jj] += prevalence.nnz
                    EpiSim.mean_incidence[jj]  += incidence.nnz
                    if track_single_outbreak:
                        EpiSim.single_cumulative[ii,jj] = cumulative.nnz # save the absolute number of infected
                        EpiSim.single_prevalence[ii,jj] = prevalence.nnz # nodes at time step jj 
                        EpiSim.single_incidence[ii,jj]  = incidence.nnz
            except (KeyboardInterrupt, SystemExit):
                print "Interrupted at: ", jj 
                print "The data will be stored in the file simulation.npz"
                EpiSim.save(fname="simulation")
                break
        
        EpiSim.mean_cumulative /= float(nodes**2)  # calculate the relative value
        EpiSim.mean_prevalence /= float(nodes**2)
        EpiSim.mean_incidence  /= float(nodes**2)
        
        if track_single_outbreak:
            EpiSim.single_cumulative /= float(nodes)  # calculate the relative value
            EpiSim.single_prevalence /= float(nodes)
            EpiSim.single_incidence  /= float(nodes)

#=====================================================================================================================
#                                            SIR Model
#=====================================================================================================================

def sir(EpiSim, memory_efficient, track_single_outbreak, save_matrix, verbose, simulation_time, infection_arrival_time, transmission_probability):
    """
    =======================================================================
    Susceptible-Infected-Model: An infected node transmits the disease
    with a probability alpha to a neighbor and recoverw after "recovery"
    time steps.
    =======================================================================
    """
    A                         = EpiSim.A
    nodes                     = A[0].shape[0]
    recovery                  = EpiSim.recovery
    memory_efficient          = bool(int(memory_efficient))
    track_single_outbreak     = bool(int(track_single_outbreak))
    save_matrix               = bool(int(save_matrix))
    verbose                   = bool(int(verbose))
    infection_arrival_time    = bool(int(infection_arrival_time))
    get_infection_time        = False
    alpha                     = float(transmission_probability)
    
    if infection_arrival_time:
        EpiSim.infection_arrival_time = {}
        infection_arrival_time = eye(nodes, nodes, 0, format='csr', dtype=float) # arrival times of source nodes = 1
        get_infection_time = True
    if verbose:
        print "=================== Initializing the simulation ==================="
        print "nodes:            ", nodes
        print "simulation time:  ", simulation_time
        print "model:            ", "sir"
        print "recovery time:    ", recovery
        print "memory efficient: ", memory_efficient
        print "track single:     ", track_single_outbreak
        print "save matrix       ", save_matrix
        print "verbose:          ", verbose
        print "infection arrival ", infection_arrival_time
    
    EpiSim.mean_cumulative = zeros((simulation_time,)) #for each time step we save the density of the 
    EpiSim.mean_prevalence = zeros((simulation_time,)) #reachability matrices, which corresponds to the mean
    EpiSim.mean_incidence  = zeros((simulation_time,)) # cumulative, prevalence and incidence
    if track_single_outbreak:
        EpiSim.single_cumulative = zeros((nodes,simulation_time)) #for each time step we save the density of the 
        EpiSim.single_prevalence = zeros((nodes,simulation_time)) #reachability matrices, which corresponds to the mean
        EpiSim.single_incidence  = zeros((nodes,simulation_time))
    
    if not memory_efficient:
        incidence_list = deque(maxlen=recovery) #records matrices, which are relevant for the infection process (first in, last out)
        for ii in xrange(recovery): # initialize the list with empty matrices...
            incidence_list.append(csr_matrix((nodes,nodes), dtype=bool))
        incidence_list.append(eye(nodes,nodes,format='csr', dtype=bool)) # ... and one unit matrix - our initial condition
        cumulative  = eye(nodes,nodes,format='csr', dtype=bool) #we keep a record of the cumulative incidence
        prevalence  = eye(nodes,nodes,format='csr', dtype=bool) #and the prevalence
        print "========================= Start the simulation ================"
        for jj in xrange(0,simulation_time):
            try:
                if verbose: #print progress only if the simulation runs on the local computer
                    if not jj % int(simulation_time/100.):
                        print 'progress: ', round(1.*jj/simulation_time*100) , ' percent , prevalence: ', prevalence.nnz
                contact_matrix = A[jj].copy()
                if not alpha == 1: # remove links if no infection is transmitted
                    contact_matrix.data *= random(contact_matrix.data.shape) < alpha
                    contact_matrix.eliminate_zeros()
                # ============= This is important ============================
                incidence   = contact_matrix.dot(prevalence) # infection transmission
                incidence  -= incidence.multiply(cumulative) # remove previously infected nodes
                prevalence -= incidence_list[0] # remove recovered nodes
                prevalence += incidence # add newly infected nodes to prevalence
                cumulative += incidence # and to the cumulative matrix
                incidence_list.append(incidence) # newly infected nodes are added and recovered nodes are removed implicitly
                # ============================================================
                if get_infection_time:
                    infection_arrival_time = update_infection_arrival_times(incidence, infection_arrival_time, jj)
                if track_single_outbreak:
                    EpiSim.single_cumulative[:,jj] = cumulative.getnnz(axis=0) # save the absolute number of infected
                    EpiSim.single_prevalence[:,jj] = prevalence.getnnz(axis=0) # nodes at time step jj 
                    EpiSim.single_incidence[:,jj]  = incidence.getnnz(axis=0)
                EpiSim.mean_cumulative[jj] = cumulative.nnz # save the absolute number of infected
                EpiSim.mean_prevalence[jj] = prevalence.nnz # nodes at time step jj 
                EpiSim.mean_incidence[jj]  = incidence.nnz
                
            except (KeyboardInterrupt, SystemExit):
                print "Interrupted at: ", jj 
                print "The data will be stored in the file simulation.npz"
                EpiSim.save(fname="simulation")
                break
        EpiSim.mean_cumulative /= float(nodes**2)  # calculate the relative value
        EpiSim.mean_prevalence /= float(nodes**2)
        EpiSim.mean_incidence  /= float(nodes**2)
        if track_single_outbreak:
            EpiSim.single_cumulative /= float(nodes)  # calculate the relative value
            EpiSim.single_prevalence /= float(nodes)
            EpiSim.single_incidence  /= float(nodes)
        
        if get_infection_time:
            EpiSim.infection_arrival_time['data'] = infection_arrival_time.data
            EpiSim.infection_arrival_time['indices'] = infection_arrival_time.indices
            EpiSim.infection_arrival_time['indptr'] = infection_arrival_time.indptr
        
        if save_matrix:
            EpiSim.data = cumulative.data
            EpiSim.indices = cumulative.indices
            EpiSim.indptr = cumulative.indptr
        
    else: # memory efficient (but slow) algorithm
        print "========================= Start the simulation ================"
        for ii in xrange(nodes):
            try:
                if verbose:
                    print 'node ', ii, ' of ', nodes
                
                seed = zeros((nodes,1), dtype=bool)
                seed[ii] = 1. # go through all nodes and take them as the seed of infection
                incidence_list = deque(maxlen=recovery)
                for kk in xrange(recovery):
                    incidence_list.append(csr_matrix((nodes,1), dtype=bool))
                incidence_list.append(csr_matrix(seed, dtype=bool))
                cumulative         = csr_matrix(seed, dtype=bool)
                prevalence         = csr_matrix(seed, dtype=bool)
                
                for jj in range(0,simulation_time):
                    contact_matrix = A[jj].copy()
                    if not alpha == 1: # remove links if no infection is transmitted
                        contact_matrix.data *= random(contact_matrix.data.shape) < alpha
                        contact_matrix.eliminate_zeros()
                    # ============= This is important ============================
                    incidence   = contact_matrix.dot(prevalence) # infection transmission
                    incidence  -= incidence.multiply(cumulative) # remove previously infected nodes
                    prevalence -= incidence_list[0] # remove recovered nodes
                    prevalence += incidence # add newly infected nodes to prevalence
                    cumulative += incidence # and to the cumulative matrix
                    incidence_list.append(incidence) # newly infected nodes are added and recovered nodes are removed implicitly
                    # ============================================================
                    EpiSim.mean_cumulative[jj] += cumulative.nnz
                    EpiSim.mean_prevalence[jj] += prevalence.nnz
                    EpiSim.mean_incidence[jj]  += incidence.nnz
                    if track_single_outbreak:
                        EpiSim.single_cumulative[ii,jj] = cumulative.nnz # save the absolute number of infected
                        EpiSim.single_prevalence[ii,jj] = prevalence.nnz # nodes at time step jj 
                        EpiSim.single_incidence[ii,jj]  = incidence.nnz
            except (KeyboardInterrupt, SystemExit):
                print "Interrupted at: ", jj 
                print "The data will be stored in the file simulation.npz"
                EpiSim.save(fname="simulation")
                break
        
        EpiSim.mean_cumulative /= float(nodes**2)  # calculate the relative value
        EpiSim.mean_prevalence /= float(nodes**2)
        EpiSim.mean_incidence  /= float(nodes**2)
        
        if track_single_outbreak:
            EpiSim.single_cumulative /= float(nodes)  # calculate the relative value
            EpiSim.single_prevalence /= float(nodes)
            EpiSim.single_incidence  /= float(nodes)
        
#=====================================================================================================================
#                                            SI Model
#=====================================================================================================================

def si(EpiSim, memory_efficient, track_single_outbreak, save_matrix, verbose, simulation_time, infection_arrival_time, transmission_probability):
    """
    =======================================================================
    Susceptible-Infected-Model: An infected node transmits the disease
    with a probability alpha to a neighbor with no recovery.
    =======================================================================
    """
    A                         = EpiSim.A
    nodes                     = A[0].shape[0]
    recovery                  = EpiSim.recovery
    memory_efficient          = bool(int(memory_efficient))
    track_single_outbreak     = bool(int(track_single_outbreak))
    save_matrix               = bool(int(save_matrix))
    verbose                   = bool(int(verbose))
    infection_arrival_time    = bool(int(infection_arrival_time))
    get_infection_time        = False
    alpha                     = float(transmission_probability)
    
    if infection_arrival_time:
        EpiSim.infection_arrival_time = {}
        infection_arrival_time = eye(nodes, nodes, 0, format='csr', dtype=float) # arrival times of source nodes = 1
        get_infection_time = True
    if verbose:
        print "=================== Initializing the simulation ==================="
        print "nodes:            ", nodes
        print "observation time: ", len(A)
        print "simulation time:  ", simulation_time
        print "model:            ", "si"
        print "recovery time:    ", recovery
        print "memory efficient: ", memory_efficient
        print "track single:     ", track_single_outbreak
        print "save matrix       ", save_matrix
        print "verbose:          ", verbose
        print "infection arrival ", infection_arrival_time
    
    EpiSim.mean_cumulative = zeros((simulation_time,)) #for each time step we save the density of the 
    EpiSim.mean_prevalence = zeros((simulation_time,)) #reachability matrices, which corresponds to the mean
    EpiSim.mean_incidence  = zeros((simulation_time,)) # cumulative, prevalence and incidence
    
    if track_single_outbreak:
        EpiSim.single_cumulative = zeros((nodes,simulation_time)) #for each time step we save the density of the 
        EpiSim.single_prevalence = zeros((nodes,simulation_time)) #reachability matrices, which corresponds to the mean
        EpiSim.single_incidence  = zeros((nodes,simulation_time)) #over all initial (seed) nodes.
    
    if not memory_efficient:
        cumulative = eye(nodes, nodes, 0, format='csr', dtype=bool)
        for ii in xrange(0,simulation_time):
            contact_matrix = A[ii].copy()
            if not alpha == 1:
                contact_matrix.data *= random(contact_matrix.data.shape) < alpha
                contact_matrix.eliminate_zeros()
            incidence = contact_matrix.dot(cumulative) # infection transmission happens here!
            if get_infection_time:
                infection_arrival_time = update_infection_arrival_times(incidence, cumulative, infection_arrival_time, ii)
            cumulative += incidence
            EpiSim.mean_cumulative[ii] = cumulative.nnz
            if track_single_outbreak:
                EpiSim.single_cumulative[:,ii] = cumulative.getnnz(axis=0) # save the absolute number of infected
            if verbose:
                print 'step ', ii, ' of ', simulation_time, ", cumulative: ", EpiSim.mean_cumulative[ii]
        EpiSim.mean_cumulative = EpiSim.mean_cumulative.astype(float)/nodes**2
        if get_infection_time:
            infection_arrival_time.setdiag(0)
            EpiSim.infection_arrival_time['data'] = infection_arrival_time.data
            EpiSim.infection_arrival_time['indices'] = infection_arrival_time.indices
            EpiSim.infection_arrival_time['indptr'] = infection_arrival_time.indptr
    else: # memory efficient algorithm
        for ii in xrange(nodes):
            if verbose:
                print 'node ', ii, ' of ', nodes
            cumulative     = zeros((nodes,1), dtype=bool)
            cumulative[ii] = 1.
            cumulative     = csr_matrix(cumulative, dtype=bool)
            for jj in xrange(1,simulation_time):
                cumulative += A[jj]*cumulative
                EpiSim.mean_cumulative[jj] += cumulative.nnz
                if track_single_outbreak:
                    EpiSim.single_cumulative[ii,jj] = cumulative.nnz # save the absolute number of infected
        EpiSim.mean_cumulative = EpiSim.mean_cumulative.astype(float)/nodes**2
    
    EpiSim.mean_prevalence = EpiSim.mean_cumulative
    EpiSim.mean_incidence  = gradient(EpiSim.mean_cumulative)
    if track_single_outbreak:
        EpiSim.single_cumulative /= float(nodes)  # calculate the relative value
        EpiSim.single_prevalence = EpiSim.single_cumulative
        EpiSim.single_incidence  = gradient(EpiSim.single_prevalence, axis=1)
# ============================================================================
#                        Tools
# ============================================================================

def update_infection_arrival_times(incidence, cumulative, infection_arrival_time, time_step):
    new_arrivals = incidence.multiply(cumulative) #element-wise "and" gives elements, which are present in both matrices
    new_arrivals -= incidence #element-wise "xor" returns elements of incidence, which are not present in cumulative, i.e. new transmission paths
    new_arrivals.data = new_arrivals.data*(time_step+1)
    infection_arrival_time += new_arrivals
    return infection_arrival_time