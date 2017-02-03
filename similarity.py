# Load modules
from spark_config import *
from operator import add
from random import uniform
from math import sqrt
from util import *

########################################################################################################################
### Subfunctions for mappers to compute similarities ###
def process_mat_row(row):
    values = row.split("\t")
    index = int(values[0])
    values = [float(_) for _ in values[1:]]
    return [[index, i, j] for i, j in enumerate(values)]

def key_var(row):
    return row[1],(row[0],row[2])

def get_distances(row):
    index, ((i, vi), (j, vj)) = row
    return (i, j), -(vi-vj)**2 #similarities = euclidian distance between two datapoints

#def get_final_distances(row):
#    (i,j),val = row
#    return (i,j), -sqrt(val)

########################################################################################################################
### Computation of the matrix of similarities ###
def similarity_matrix(file):
    """
    Returns a matrix of similarities between observations, given an input matrix in Spark format
    :param file containing datapoints
    :return: RDD object containing similarities. Each row is indexed by (index i individual 1, index j individual 2)
    and the value is the similarity between individuals i and j.
    """
    # load input datafile
    mat = sc.textFile(file)
    # convert to Spark format: all variables are separated. Index is num of the variable (= num of column).
    # mat containing datapoints in spark format : (individual index, variable index), value
    new_mat = mat.flatMap(process_mat_row)
    # join the matrix with itself, to have the combination of each point with all the other ones
    # variable per variable (regarding index)
    mat_join = new_mat.map(key_var).join(new_mat.map(key_var))
    # use a mapper to compute squared euclidian distance for each variable between all individuals
    #mat_sim = mat_join.map(get_var_sqdistances).reduceByKey(add).map(get_final_distances)
    mat_sim = mat_join.map(get_distances).reduceByKey(add)
    return mat_sim

########################################################################################################################
### Subfunctions to compute preferences ###
def filter_preferences(row):
    (i,j),s=row
    if (i!=j):
        return row

def sum_similarities(row):
     i, values = row
     return i,sum([c[1] for c in values])

def get_max_dpsim2(row):
    k,((i,si),(j,sj))=row
    if (i!=j):
        return (i,j),max(si,sj)

def replace_preferences_mapper(row, Pstep):
    (i, j), s = row
    if (i != j):
        return row
    else:
        return (i,j), Pstep

########################################################################################################################
### Main functions to compute preferences ###
def get_Pmax(S):
    Pmax=S.map(filter_preferences).filter(lambda x: x is not None).max()
    return Pmax[1]

def get_Pmin(S):
    ## get dpsim1
    dpsim1=reformate_RDD(S).map(sum_similarities).max()
    # print(dpsim1)
    ## get dpsim2
    dpsim2=S.map(index_to_value).join(S.map(index_to_value)).map(get_max_dpsim2).filter(lambda x: x is not None).reduceByKey(add).max()
    # print(dpsim2)
    ## compute Pmin
    Pmin = dpsim1[1]-dpsim2[1]
    return Pmin

def get_Pstep(Pmin,Pmax,K,N):
    return (Pmax-Pmin)/(N*0.1*sqrt(K+50))

def integrate_preferences(S,Pstep):
    return S.map(lambda row: replace_preferences_mapper(row,Pstep))