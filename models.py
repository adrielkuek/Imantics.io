# models 
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# for everything else
import os, pickle, pprint
import numpy as np
import pandas as pd

    ################ PCA, KMeans, KNN ##################
    ##knn_neighbors = 20

def rename_clusters(data_groups):
    data_groupings = {}
    data_filenames = {}
    for key, value in data_groups.items():
        groupname = int(key.replace("cluster_",""))
        data_groupings[groupname] = value

    data_dict = dict(sorted(data_groupings.items()))
    
    for key in data_dict.keys():
        if key < 10:
            data_filenames["Cluster_00" + str(key)] = data_dict[key]
        elif key >= 10 and key < 100:
            data_filenames["Cluster_0" + str(key)] = data_dict[key]
        else:
            data_filenames["Cluster_" + str(key)] = data_dict[key]

    return data_filenames

def Dino_250():

    general_path = os.getcwd()
    with open(general_path +'\\feature_tensors\\DINO\\DINO_250clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups)

    return data_filenames

def Dino_200():

    general_path = os.getcwd()
    with open(general_path +'\\feature_tensors\\DINO\\DINO_200clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups)

    return data_filenames

def Dino_300():

    general_path = os.getcwd()
    with open(general_path +'\\feature_tensors\\DINO\\DINO_300clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups)

    return data_filenames

def Dino_350():

    general_path = os.getcwd()
    with open(general_path +'\\feature_tensors\\DINO\\DINO_350clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups)

    return data_filenames

def Dino_400():

    general_path = os.getcwd()
    with open(general_path +'\\feature_tensors\\DINO\\DINO_400clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups)

    return data_filenames

def Dino_450():

    general_path = os.getcwd()
    with open(general_path +'\\feature_tensors\\DINO\\DINO_450clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups)

    return data_filenames

def Scan_200():

    general_path = os.getcwd()
    with open(general_path +'\\feature_tensors\\SCAN\\SCAN_200clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups)

    return data_filenames

##    ################# Train Test Split #################
##    Y = kmeans.labels_.tolist()
##    X_train, X_test, y_train, y_test = train_test_split(DTtensors, Y, test_size=0.6, random_state = 42)
##    knn = KNeighborsClassifier(knn_neighbors)
##    knn.fit(X_train, y_train)
##    pred =  knn.predict(X_test)
##
##    ################### Combine with filenames ##########
##    filename =[]
##    num = 1
##    for file in data_filenames:
##        filename.append([num,file])
##        num += 1
##    filename = pd.DataFrame(filename, columns=["index","filename"])
##
##    prediction = pred.tolist()
##    prediction_data = X_test
##    prediction_data["label"] = prediction
##    prediction_data["index"] = prediction_data.index
##
##    inner_join = pd.merge(prediction_data, filename, on ='index', how ='inner')
##
##    ################### Dictionary of cluster & filenames ####
##
##    groups = {}
##    renamed_groups = {}
##    for cluster, file in zip(inner_join["label"],inner_join["filename"]):
##    ##    if cluster < 10:
##    ##        cluster_name = "Group_0" + str(cluster)
##    ##    else:
##    ##        cluster_name = "Group_0" + str(cluster)
##        
##        if cluster not in groups.keys():
##            groups[cluster] = []
##            groups[cluster].append(file)
##        else:
##            groups[cluster].append(file)
##    groupings = dict(sorted(groups.items()))
##
##    for key in groupings.keys():
##        if int(key) < 10:
##            renamed_groups["Group_0" + str(key)] = groupings[key]
##        else:
##            renamed_groups["Group_" + str(key)] = groupings[key]

    



