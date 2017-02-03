from util import *
import time

def dampen_message(row,lamb):
    (i,j),(new,old)=row
    return (i,j),lamb*old+new*(1-lamb)

def dampen_without_old(row,lamb):
    (i,j),new=row
    return (i,j),new*(1-lamb)

#### Responsibilities ####
def get_max_by_j(row):
    # used to compute the second term in equation (2)
    # return output indexed by i
    # output is a vector of the second term of equation (2) for each k
    # - max (a(i,j)+s(i,j)) for j != k
    i,iterable = row
    values=list(iterable)
    output=[]
    for j in list(values):
        output.append((j[0],-max([c for c in values if c[0]!=j[0]],key= lambda x: x[1])[1]))
    return i,output

def update_responsibility(S,A,R_prev,lamb=0.5):
    """ update responsibility and dampen messages with parameter lamb as described in the article """
    avsim_max = reformate_RDD(S.join(A).map(lambda row: (row[0],sum(row[1])))).map(get_max_by_j).flatMap(redispatch)
    R_new=S.join(avsim_max).map(sum_terms)
    if R_prev is not None:
        R=R_new.join(R_prev).map(lambda row: dampen_message(row,lamb))
    else:
        R=R_new.map(lambda row: dampen_without_old(row,lamb))
    return R

#### Availabilities ####
def get_max_by_ij(row):
    # used to get vi in equation (3), in function get_min_to_avail_update
    i,iterable = row
    values=list(iterable)
    output=[]
    # return output indexed by i
    # output is a vector of the second term of equation (2) for each k
    # - max (a(i,j)+s(i,j)) for j != k
    for j in values:
        output.append((j[0],-max([c for c in values if (c[0]!=j[0]) & (c[0]!=i)],key= lambda x: x[1])[1]))
    return i,output

def inverse_indexes(row):
    # to inverse i and k in the key (i,k), indexing values like r(i,k) or a(i,k)
    (i,j),values=row
    return (j,i), values

def filter_self(row):
    # used to filter self responsibilities, used in computation of equation (3)
    (i,j),values=row
    if i==j:
        return i,values

def get_min_to_avail_update(row):
    # as in equation (3), return min of 0 and (vi+r) for non self-availabilities
    # return vi for self-availabilities
    k,((i,vi),r)=row
    if i!=k:
        return (i,k),min(0,vi+r)
    else:
        return (i,k),vi

def update_availability(R,A_prev,lamb=0.5):
    """ update availability and dampen messages with parameter lamb as described in the article """
    ### equations (3) and (4) ###
    sum_rmax=reformate_RDD(R.map(inverse_indexes)).map(get_max_by_ij).flatMap(redispatch).map(index_to_value)
    # get self-responsibility to calculate the final value
    self_resp=R.map(filter_self).filter(lambda x: x is not None)
    # join the two elements to add = r(k,k) + sum(max(...)) (see equation 3) and return min(0,r(k,k)+...)
    new_A=sum_rmax.join(self_resp).map(get_min_to_avail_update)
    # dampen messages to avoid numerical oscillations
    A=new_A.join(A_prev).map(lambda row: dampen_message(row,lamb))
    return A