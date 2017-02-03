import pandas
import numpy as np
import math
import random


def similatry_matrix(A):
    x=np.shape(A)
    B=np.zeros(shape=(x[0],x[0]))
    for i in range(x[0]):
        for j in range(x[0]):
            if i%1000==0 and j ==x[0]-1:
                print('Iteration Similarity',i)
            B[i,j]=-np.linalg.norm(A[i,:]-A[j,:])**2
    return(B)

    
def preference_range(SM):
    mask = np.ones(SM.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    pmax = SM[mask].max()
    dpsim1=np.amax(np.sum(SM,axis=1))
    x=np.shape(SM)
    dpsim2=0
    print('Minima calculations')
    for i in range(x[0]-1):
        for j in range(i+1,x[0]) :
            if i==0 and j==0:
                dpsim2=np.sum(np.max(np.vstack((SM[i,:],SM[j,:])),axis=0))
            if i%1000==0 and j==x[0]-1:
                print("Iteration individual",i)
            else:
                dpsim2=max(dpsim2,np.sum(np.max(np.vstack((SM[i,:],SM[j,:])),axis=0)))
    return(pmax,dpsim1-dpsim2,(pmax-(dpsim1-dpsim2))/(x[0]*0.1*math.sqrt(x[0]+50)))

def update_responsabilities(A,SM):
    R=np.zeros(SM.shape)
    n=np.shape(SM)[0]
    for i in range(n):
        for k in range(n):
            mask=np.ones(n,dtype=bool)
            mask[i]=0
            mask[k]=0
            R[i,k]=SM[i,k] - (A[i,:]+SM[i,:])[mask].max()
    return(R)
            
def update_avaibility(R,SM):
    A=np.zeros(SM.shape)
    Zeros=np.zeros(SM.shape)
    n=np.shape(SM)[0]
    for i in range(n):
        for k in range(n):
            mask=np.ones(n,dtype=bool)
            mask[i]=0
            mask[k]=0
            A[i,k]=min(0,R[k,k] + (np.maximum(Zeros[i,:],SM[i,:]))[mask].sum())
    for i in range(n):
        mask=np.ones(n,dtype=bool)
        mask[i]=0
        A[i,i]=np.sum(np.maximum(Zeros,A)[i,:][mask])
    return(A) 


def center_matrix(A,R):
    x=np.shape(A)
    mask=np.ones(x,dtype=bool)
    np.fill_diagonal(mask, 0)
    U=(A+R)
    return(np.argmax(U,axis=1))

def Final_Boucle(B,nb_iter):
    print('Similarity')
    SM=similatry_matrix(B)
    pref=SM.min()
    np.fill_diagonal(SM,pref)
    n=np.shape(B)[0]
    A=np.zeros((n,n))
    R=np.zeros((n,n))
    print('Beginning of the Loop')
    print('number of loops:',nb_iter)
    for i in range(nb_iter):
        #if i%10 ==0:
        print('Iteration Responsability',i)
        R=update_responsabilities(A,SM)
        #if i%10 ==0:
        print('Iteration Avaibility',i)
        A=update_avaibility(R,SM)
    print('End of Loop, searching center of clusters')
    Centers=center_matrix(A,R)
    return Centers


############ FUNCTIONS FOR MAPPERS ###############
def go_clustering(B,nb_iter,S):
    n=np.shape(B)[0]
    A=np.zeros((n,n))
    R=np.zeros((n,n))
    print('Beginning of the Loop')
    print('number of loops:',nb_iter)
    for i in range(nb_iter):
        #if i%10 ==0:
        print('Iteration Responsability',i)
        R=update_responsabilities(A,S)
        #if i%10 ==0:
        print('Iteration Avaibility',i)
        A=update_avaibility(R,S)
    print('End of Loop, searching center of clusters')
    Centers=center_matrix(A,R)
    return Centers

# functions for algorithm
def define_subgroup(row):
    i,values=row
    group=random.randint(1,4) # on a 4 coeurs
    return group,(i,values)

def index_values(row):
    values = row.split("\t")
    index = int(values[0])
    values = [float(_) for _ in values[1:]]
    return index,values

def do_AP(row,SM):
    group,iterable=row
    values=list(iterable) # from python iterable to list
    indices=[c[0] for c in values]
    matrice=np.array([v[1] for v in values])
    S=np.array(pandas.DataFrame(SM).ix[indices,indices])
    centers = go_clustering(matrice,50,S)
    return [(indices[j],indices[i]) for i,j in enumerate(centers)] # key is the exemplar index
        
        

          