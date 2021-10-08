# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os, pickle, PIL, torch, pprint
import numpy as np
from random import randint
import pandas as pd

def CLIP():
    ################ PCA, KMeans, KNN ##################
    pca_components = 100
    cluster_size = 50
    knn_neighbors = 5

    ################ Initialization ####################
    general_path = os.getcwd()
    with open(general_path +'\\CLIP OurDataSet Tensors\\image_filenames_OurDataset.pkl', 'rb') as f:
        data_filenames = pickle.load(f)

    embedding_file = general_path + "\\CLIP OurDataSet Tensors\\image_embeddings_OurDataset.pt"
    features_tensors = torch.load(embedding_file, map_location=torch.device('cpu'))
    DTtensors = pd.DataFrame(features_tensors.numpy(), index= range(1,len(features_tensors)+1))

    ################ Dimensionality Reduction ##########
    pca = PCA(n_components=pca_components, random_state=22)
    pca.fit(features_tensors)
    x = pca.transform(features_tensors)

    eigenvalues = pca.explained_variance_
    eigenvectors = np.round(pca.components_.transpose(),decimals=3)
    var_expln = np.round(pca.explained_variance_ratio_ * 100,decimals=3)
    loadings = eigenvectors * np.sqrt(eigenvalues)

    ################# KMeans Clustering ################
    kmeans = KMeans(n_clusters=cluster_size, random_state=22)
    kmeans.fit(x)

    ################# Train Test Split #################
    Y = kmeans.labels_.tolist()
    X_train, X_test, y_train, y_test = train_test_split(DTtensors, Y, test_size=0.6, random_state = 42)
    knn = KNeighborsClassifier(knn_neighbors)
    knn.fit(X_train, y_train)
    pred =  knn.predict(X_test)

    ################### Combine with filenames ##########
    filename =[]
    num = 1
    for file in data_filenames:
        filename.append([num,file])
        num += 1
    filename = pd.DataFrame(filename, columns=["index","filename"])

    prediction = pred.tolist()
    prediction_data = X_test
    prediction_data["label"] = prediction
    prediction_data["index"] = prediction_data.index

    inner_join = pd.merge(prediction_data, filename, on ='index', how ='inner')

    ################### Dictionary of cluster & filenames ####

    groups = {}
    renamed_groups = {}
    for cluster, file in zip(inner_join["label"],inner_join["filename"]):
    ##    if cluster < 10:
    ##        cluster_name = "Group_0" + str(cluster)
    ##    else:
    ##        cluster_name = "Group_0" + str(cluster)
        
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    groupings = dict(sorted(groups.items()))

    for key in groupings.keys():
        if int(key) < 10:
            renamed_groups["Group_0" + str(key)] = groupings[key]
        else:
            renamed_groups["Group_" + str(key)] = groupings[key]


    ##pprint.pprint(renamed_groups)

    return renamed_groups

