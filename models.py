# models 
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# for everything else
import os, pickle, pprint
import numpy as np
import pandas as pd
import statistics

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


def stats(clustersize, data_filenames):
    dict_length = {}
    for key,value in data_filenames.items():
        dict_length[key] = len(value)
    imgcount_nonzero = len(data_filenames) 
    imgcount_zero = clustersize - imgcount_nonzero
    max_value = max(dict_length.values())
    min_value = min(dict_length.values())
    mean_value = round(statistics.mean(dict_length.values()),2)
##    print(imgcount_zero)
##    print(imgcount_nonzero)
##    print(max_v)
##    print(min_v)
##    print(mean_v)

    return imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value

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
   
    


