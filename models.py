# models 
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.cluster import normalized_mutual_info_score,rand_score, adjusted_rand_score

# for everything else
import os, pickle, pprint, sys
import numpy as np
import pandas as pd
import statistics
from pathlib import Path

def rename_clusters(data_groups,label):
    data_groupings = {}
    data_filenames = {}
    for key, value in data_groups.items():
        groupname = int(key.replace(label,""))
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

    return imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value

def Dino_50():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_KMeans/DINO_50clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Dino_100():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_KMeans/DINO_100clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Dino_150():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_KMeans/DINO_150clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Dino_200():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_KMeans/DINO_200clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Dino_250():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_KMeans/DINO_250clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Dino_300():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_KMeans/DINO_300clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Dino_350():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_KMeans/DINO_350clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Dino_400():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_KMeans/DINO_400clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Dino_450():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_KMeans/DINO_450clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Dino_500():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_KMeans/DINO_500clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Scan_200():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/SCAN/SCAN_200clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")

    return data_filenames

def Vgg_200():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/VGG/VGG16_cluster.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"Cluster_")
    
    return data_filenames
    
def Kmplus_Dino_50():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Kmeansplus/DINO_Kmeansplus_50clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Kmplus_Dino_100():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Kmeansplus/DINO_Kmeansplus_100clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Kmplus_Dino_150():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Kmeansplus/DINO_Kmeansplus_150clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Kmplus_Dino_250():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Kmeansplus/DINO_Kmeansplus_250clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Kmplus_Dino_200():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Kmeansplus/DINO_Kmeansplus_200clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Kmplus_Dino_300():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Kmeansplus/DINO_Kmeansplus_300clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Kmplus_Dino_350():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Kmeansplus/DINO_Kmeansplus_350clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Kmplus_Dino_400():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Kmeansplus/DINO_Kmeansplus_400clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Kmplus_Dino_450():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Kmeansplus/DINO_Kmeansplus_450clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Kmplus_Dino_500():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Kmeansplus/DINO_Kmeansplus_500clusters.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Agglo_Dino_11():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Agglo/DINO_agglo_dist1_1_wardlinkage.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Agglo_Dino_15():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Agglo/DINO_agglo_dist1_5_wardlinkage.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Agglo_Dino_18():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Agglo/DINO_agglo_dist1_8_wardlinkage.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Agglo_Dino_25():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Agglo/DINO_agglo_dist2_5_wardlinkage.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Agglo_Dino_30():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Agglo/DINO_agglo_dist3_0_wardlinkage.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def DBSCAN_Dino_EP8_M2():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Dbscan/DINO_dbscan_eps08_minsample2.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def DBSCAN_Dino_EP8_M3():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Dbscan/DINO_dbscan_eps08_minsample3.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def DBSCAN_Dino_EP9_M2():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Dbscan/DINO_dbscan_eps09_minsample2.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Optics_Dino_M3():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_optics/DINO_optics_minsample3.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Optics_Dino_M4():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_optics/DINO_optics_minsample4.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Optics_Dino_M5():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_optics/DINO_optics_minsample5.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Agglo_Dino_Ed12():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Agglo_ED/DINO_agglo_eucldist1_2.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Agglo_Dino_Ed15():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Agglo_ED/DINO_agglo_eucldist1_5.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames

def Agglo_Dino_Ed20():

    general_path = os.getcwd()
    with open(general_path +'/feature_tensors/DINO_Agglo_ED/DINO_agglo_eucldist2_0.pkl', 'rb') as f:
        data_groups = pickle.load(f)

    data_filenames = rename_clusters(data_groups,"cluster_")
    
    return data_filenames
