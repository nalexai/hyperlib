import numpy as np
import networkx as nx
from hyperlib.embedding.treerep import treerep
from hyperlib.embedding.sarkar import sarkar_embedding
from hyperlib.utils.multiprecision import poincare_dist

# Sarich measured "immunological distances" between 8 mammal species.
# This metric should be hyperbolic because it is related to the species' 
# distance on the evolutionary tree. 
#
# For more details on hyperbolicity see https://meiji163.github.io/post/combo-hyperbolic-embedding/ 

mammals = ["dog", "bear", "raccoon", "weasel", "seal", "sea_lion", "cat", "monkey"]
labels = {i: m for i, m in enumerate(mammals)}
compressed_metric = np.array([ 
        32.,  48.,  51.,  50.,  48.,  98., 148.,  
        26.,  34.,  29.,  33., 84., 136.,  
        42.,  44.,  44.,  92., 152.,  
        44.,  38.,  86., 142.,
        42.,  89., 142.,  
        90., 142., 
        148.
    ])

# Using TreeRep we can construct a putative evolutionary tree 

tree = treerep(compressed_metric, return_networkx=True) # outputs a weighted networkx Graph
nx.draw(tree, labels=labels, with_labels=True) #plot tree

# Using Sarkar's algorithm we can embed the tree in the Poincare ball

root = 0 # label of root node
tau = 0.2 # scaling factor for edges
embed_2D = sarkar_embedding(tree, root, tau=tau)

# calculate hyperbolic distances from the embedding
poincare_dist(embed_2D[0,:], embed_2D[1,:])
