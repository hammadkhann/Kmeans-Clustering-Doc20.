# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 02:31:50 2017

@author: hammadkhan
"""
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from pyclust import BisectKMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes
    
    :param clusters: the cluster assignments array
    :type clusters: numpy.array
    
    :param classes: the ground truth classes
    :type classes: numpy.array
    
    :returns: the purity score
    :rtype: float
    """
    
    A = np.c_[(clusters,classes)]
    
    print((A[0:5]))

    n_accurate = 0.

    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]

# Loading documents 
doc = []
#list of files
filenames = os.listdir("Doc50")
for i in range(50):
    with open("Doc50/"+filenames[i], 'r') as documents:
        doc.append(documents.read())
        documents.close()
        
#==============================================================================
# for i in range(0, 50):
#     doc[i] = re.sub('[^a-zA-Z]', ' ', doc[i])
#     doc[i] = doc[i].lower()
#     doc[i] = doc[i].split()
#     ps = PorterStemmer()
#     doc[i] = [ps.stem(word) for word in doc[i] if not word in set(stopwords.words('english'))]
#     doc[i] = ' '.join(doc[i])
#     
#==============================================================================


#==============================================================================
# #Bag of words feature extraction
# count_vect = CountVectorizer(ngram_range = (1,3), encoding='utf-8',decode_error='replace',strip_accents='unicode'
#                   ,analyzer='word',max_df= 0.25, min_df = 7, max_features = 25) 
# count_vectorized = count_vect.fit_transform(doc)
# 
# count_vocab = count_vect.get_feature_names()
#==============================================================================

#==============================================================================
# from sklearn.decomposition import TruncatedSVD
# svd = TruncatedSVD(n_components=1, n_iter=15, random_state=0)
# svd.fit(count_vectorized)  
# svd.transform(count_vectorized)
# 
#==============================================================================


#Tfidf features
tfidf_vect = TfidfVectorizer(ngram_range = (1,3),norm='l2',smooth_idf = False , analyzer='word', max_df= 0.25,min_df = 9, stop_words = 'english')
tfidf_vectorized = tfidf_vect.fit_transform(doc)
count_vocab = tfidf_vect.get_feature_names()
#Stacking
#count_vectorized = hstack((count_vectorized,tfidf_vectorized))
count_vectorized = tfidf_vectorized
#cluster range
range_n_clusters = [5]

for n_clusters in range_n_clusters:
    #initializing models
    clusterer_k = KMeans(n_clusters=n_clusters, random_state=99)
    clusterer_a = AgglomerativeClustering(n_clusters=n_clusters)
    #clusterer_b = BisectKMeans(n_clusters=n_clusters)
    
    #fitting and predicting the clusters
    cluster_labels_k = clusterer_k.fit_predict(count_vectorized.toarray())
    cluster_labels_a = clusterer_a.fit_predict(count_vectorized.toarray())
    #cluster_labels_b = clusterer_b.fit_predict(count_vectorized.toarray())
    
    #Calculating Silhouette average scores for each cluster
    silhouette_avg = silhouette_score(count_vectorized, cluster_labels_k)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score_k is :",  silhouette_avg)
    
    silhouette_avg = silhouette_score(count_vectorized, cluster_labels_a)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score_a is :", silhouette_avg)

gt = []
gt += 10 * [4]
gt += 10 * [0]
gt += 10 * [1]
gt += 10 * [2]
gt += 10 * [3]

gt = np.array(gt)

print(purity_score(cluster_labels_k,gt))
print(purity_score(cluster_labels_a,gt))
        
sorted_index_centroids = clusterer_k.cluster_centers_.argsort()[:,
::-1]
for i in range(5):
 print("Cluster %d words:" % i, end='')
 #replace 10 with n words per cluster
 for ind in sorted_index_centroids[i, :3]:
     print(' %s' % count_vocab[ind].split(' ')[0], end=',')
     print() #add whitespace
     print() #add whitespace

#==============================================================================
#     silhouette_avg = silhouette_score(count_vectorized, cluster_labels_b)
#     print("For n_clusters =", n_clusters,
#           "The average silhouette_score_b is :", silhouette_avg)
#==============================================================================


#Computimg the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(count_vectorized, cluster_labels_a)
y_lower = 10
for i in range(5):
    ith_cluster_silhouette_values = \
    sample_silhouette_values[cluster_labels_a == i]
ith_cluster_silhouette_values.sort()
size_cluster_i = ith_cluster_silhouette_values.shape[0]
y_upper = y_lower + size_cluster_i



sorted(filenames)
sorted(cluster_labels_k)
new_dict = dict(zip(filenames, cluster_labels_k))
print(new_dict)

