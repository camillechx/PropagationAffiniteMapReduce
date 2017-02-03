from cluster_pure_python import *
from cluster import *
import matplotlib.pyplot as plt
from scipy import misc

##### DATA #############################
## Open and plot image ##
face = misc.imread('Images/chat.png')
face=face[90:150,60:140] # on prend seulement une partie de l'image pour avoir un jeu plus petit en test
plt.imshow(face, cmap=plt.cm.gray, vmin=30, vmax=200)

dims=face.shape
# on reformate de manière à avoir une suite de points qui forment une suite de lignes
# on pourra reconstituer l'image puisque l'on connait les dimensions et que l'ordre des points est conservé
data=np.reshape(face,(dims[0]*dims[1],dims[2]))
df=pandas.DataFrame(data)
df.to_csv("face_dataset.txt", sep="\t",header=None,index=True)
print(dims)

########################################
## TEST CODE FULL MAPREDUCE ##
# avec matrice calculée par l'algo
results=algo_cluster("face_dataset.txt",dims[0]*dims[1],nb_iter=10,pref_update=False)
# échec, on passe au code suivant
# avec matrice fournie en input
# on recopie les boucles de l'algo directement pour ne pas modifier la fonction dans le fichier source py
# on ajoute des cache() pour tenter de résoudre le problème des itérations
SM=similatry_matrix(data) # fonction en python et non pas MapReduce
# on fixe une préférence pour tous
pref=random.randrange(-10,-500,-1)
print(pandas.DataFrame(SM).head())
np.fill_diagonal(SM,pref)
mat_sim=pandas.DataFrame(SM)
mat_sim.to_csv("mat_sim.txt", sep="\t",header=None,index=True)
mat_sim.head()
S=sc.textFile("mat_sim.txt").flatMap(process_mat_row).map(lambda x: ((x[0],x[1]),x[2]))
A=S.map(initialize_avail)
A.cache()
R=None
nb_iter=5
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
    R.cache()
    A=update_availability(R,A)
    A.cache()
    print("R and A updated",flush=True)
    time.sleep(1)
    time_total=time.time()-time_dep
    print(str(time_total)+" sec",flush=True)
    time.sleep(1)

print("Computing clusters...",flush=True)
time.sleep(1)
clusters=get_clusters(R,A)
clusters.cache()
print("Computed.",flush=True)
time.sleep(1)
# encore un échec, on passe à la deuxième partie du projet

########################################
## PROPAGATION D'AFFINITE DISTRIBUEE ##

# MATRICE DE SIMILARITES #
SIM=similatry_matrix(data)
pref=SIM.min()
np.fill_diagonal(SIM,pref)
mat_sim=pandas.DataFrame(SIM)
mat_sim.to_csv("cat_sim.txt", sep="\t",header=None,index=True)

# clustering : mappers #
rdd=sc.textFile("face_dataset.txt")\
	.map(index_values)\
	.map(define_subgroup)\
	.groupByKey()\
	.flatMap(lambda x: do_AP(x,SIM))\
	.groupByKey()
clusters=rdd.collect()

# test clusters given by mappers
new_data=data.copy()
for element in clusters:
    index_center = element[0]
    data_center=new_data[index_center]
    index_points = list(element[1])
    for point in index_points:
        new_data[point]=data_center

new_data=np.reshape(new_data,(dims[0],dims[1],dims[2]))
plt.imshow(new_data, cmap=plt.cm.gray, vmin=30, vmax=200)
# no changes, go to reducers to combine clusters that are neighboors

# functions for reducers
def pair_key(row):
    i,j=row
    return (i,j),0

def find_near_centers(row,pref):
    (i,j), nu, s = row
    if s < 0.5*pref:
        return i,j

def give_cluster_indice(row,clusters_exemplars):
	i,values=row
	cluster_ind=[j for j,v in enumerate(clusters_exemplars) if i in v][0]
	values.append(i)
	return cluster_ind,values # add the center to the cluster list of points


# gather neighboring clusters
exemplars = [c[0] for c in clusters]
exemplars = sc.parallelize(exemplars).map(lambda row: (0,row))

mat_sim = sc.textFile("mat_sim.txt").flatMap(process_mat_row).map(lambda x: ((x[0],x[1]),x[2])) 

near_exemplars =exemplars.join(exemplars)\
        .filter(lambda row: (row[1][0]!=row[1][1]))\
        .map(lambda row: (row[1][0],row[1][1]))\
        .map(pair_key)\
        .join(mat_sim)\
        .mapValues(lambda x: (x[0]+x[1]<0.5*pref))
        
near_exemplars.take(10) # in our application, no neighboring centers

# no results for our example with that threshold, but here's the code we produced to continue the treatment
# /!\ could not be tested #
if near_exemplars.count()>0:
    list_clusters_ex=near_exemplars.filter(lambda x: x)\
                    .map(lambda row: (row[0][0],row[1][0]))\
                    .groupByKey()\
                    .map(lambda row: sorted(list(row[0])+list(row[1])))
    # returns a RDD of lists of clusters' exemplars that are neighboring
    new_clusters=list_clusters_ex.collect()
	
	# each list of indexes gives
	final_list = set(new_clusters) # we eliminate lists that contains the same exemplars, for example [j,k,l] and [k,j,l] give the same cluster
	
	final_clusters=sc.parallelize(clusters)\
					.map(lambda x: give_cluster_indice(x,final_list))\
					.reduceByKey(add)\
					.collect()
	# we get a RDD with key = index of the cluster, and list of points in the cluster
	# we collect it, we obtain a list with same features

	# replace colors in data with color of one exemplar (by default the first) of the cluster
	final_data=data.copy()
	for cluster in final_clusters:
		ind_cluster=cluster[0]
		indice_exemplar=final_list[ind_cluster][0]
		color_cluster=data[indice_exemplar]
	    for point in cluster[1]:
	        final_data[point]=color_cluster

	final_data=np.reshape(final_data,(dims[0],dims[1],dims[2]))
	plt.imshow(final_data, cmap=plt.cm.gray, vmin=30, vmax=200)