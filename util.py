from operator import add

def index_to_value(row):
    (i,j),value = row
    return i,(j,value)

def value_to_index(row):
    i,(j,value)=row
    return (i,j),value

def redispatch(row):
    """ redispatch values of line i, [(j,value)] in multiple lines (i,j),value """
    i,values=row
    return [((i,value[0]),value[1]) for value in values]

def reformate_RDD(rdd):
    """from a key-value RDD with key (index i, index j) and value the tensor,
    returns a key-value RDD with key=index i and values as a list of (index j, value v) with v value of
    tensor indexed by (i,j) in the original RDD)"""
    rdd2=rdd.map(index_to_value)
    rdd3=rdd2.groupByKey()
    return rdd3

def sum_terms(row):
    ### sum terms of the tuple of values
    (i,j),(v1,v2)=row
    return (i,j),v1+v2
