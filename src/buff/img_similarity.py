import numpy as np
import scipy as sp
from numpy import linalg as LA
from scipy.spatial.distance import pdist

#from sklearn.metrics import hinge_loss

def data_gen():
    return img_list

def img_embedding(img):
    # W: the parameters we want to learn
    
    return np.dot(W, img)

def img_distance(img1, img2):
    f1 = img_embedding(img1)
    f2 = img_embedding(img2)
    fd = f1 - f2
    img_sim = pdist(fd, 'euclidean')
    return img_sim

def triplet_hinge_loss(g, img, img_pos, img_neg):
    # hinge loss -> a convex approximation
    fd_pos = img_distance(img, img_pos)
    fd_neg = img_distance(img, img_neg)
    return max(0, g + fd_pos - fd_neg)
    
def cost_func(esp=[triplet_hinge_loss(g, img, img_pos, img_neg) for (img, img_pos, img_neg) in data_gen()], lambda_reg=0.001 , W):
    

    return sum(esp) + lambda_reg * LA.norm(W, 2)
    
## Approximate Nearest Neighbor Search Algorithms
#  https://www.google.com/search?q=approximate+nearest+neighbor+python&oq=approximate+near&aqs=chrome.2.69i57j0l5.13040j0j7&sourceid=chrome&es_sm=93&ie=UTF-8
#  http://nearpy.io/
#  http://scikit-learn.org/stable/modules/neighbors.html
#  https://github.com/spotify/annoy
#  https://pypi.python.org/pypi/scikits.ann
#  http://www.cs.umd.edu/~mount/ANN/


    
    
