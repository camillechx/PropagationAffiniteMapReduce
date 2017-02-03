from similarity import *
from updates import *
import time

def initialize_avail(row):
    (i,j),value=row
    return (i,j),0

def find_point_cluster(row):
    i,iterable=row
    values=list(iterable)
    max_indice=values[values.index(max(values, key=lambda x: x[1]))][0]
    return i,max_indice

def get_clusters(R,A):
    """choose clusters according to responsibilities et availabilities"""
    clusters=reformate_RDD(R.join(A).map(sum_terms)).map(find_point_cluster)
    return clusters

def algo_cluster(file,N,nb_iter=10, pref_update=True):
    print("Clustering in map_reduce",flush=True)
    time.sleep(1)
    ### Initialisation ###
    # compute similarity matrix
    S=similarity_matrix(file)
    S.cache()
    print("Similarity matrix computed",flush=True)
    time.sleep(1)
    #N=sqrt(S.keys().distinct().count()) # number of individuals
    K=N # initial number of clusters = number of observations
    print("Initializing preferences...",flush=True)
    time.sleep(1) 
    Pmin=get_Pmin(S)
    Pmax=get_Pmax(S)
    # compute Pstep
    Pstep=get_Pstep(Pmin,Pmax,K,N)
    # integrate preferences
    S=integrate_preferences(S,Pstep)
    print("Preferences initialized.",flush=True)
    time.sleep(1) 
    # initialize availability and responsibility
    A=S.map(initialize_avail)
    R=None
    print("Matrices initialized")
    ### Iterations ###
    for n in range(0,nb_iter):
        print("----------------------------------------------",flush=True)
        time.sleep(1)
        time_dep=time.time()
        print("Itération "+str(n+1),flush=True)
        time.sleep(1) 
        # update responsibilty and availability
        R=update_responsibility(S,A,R)
        #print(R.distinct().count(),flush=True)
        #time.sleep(1)
        A=update_availability(R,A)
        #print(A.distinct().count(),flush=True)
        #time.sleep(1)
        print("R and A updated",flush=True)
        time.sleep(1) 
        
        if pref_update:
            # find clusters
            print("Computing clusters...",flush=True)
            time.sleep(1)
            clusters=get_clusters(R,A)
            print("Clusters updated.",flush=True)
            time.sleep(1)
            # update Pstep regarding number of clusters
            print("Updating preferences...",flush=True)
            time.sleep(1)
            K=clusters.distinct().count()
            print(K,flush=True)
            time.sleep(1)
            Pstep=get_Pstep(Pmin,Pmax,K,N)
            # update preferences in the similarity matrix
            S = integrate_preferences(S, Pstep)
            print("Preferences updated",flush=True)
            time.sleep(1)
            
        time_total=time.time()-time_dep
        print(str(time_total)+" sec",flush=True)
        time.sleep(1)
        
    if not pref_update:
        print("Computing clusters...",flush=True)
        time.sleep(1)
        clusters=get_clusters(R,A)
        print("Computed.",flush=True)
        time.sleep(1)
        
    return clusters