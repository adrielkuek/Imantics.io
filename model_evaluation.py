from sklearn.metrics.cluster import normalized_mutual_info_score,rand_score, adjusted_rand_score
import os
import pandas as pd
import pickle, sys
import pprint
from pathlib import Path

def extract_labels(model_file, filename):
    img_list, cluster_list, imgno_list = [], [], []
    for key, value in model_file.items():
        for img in value:
            img_list.append(img)
            cluster_list.append(key.replace("cluster_",""))
            imgno_list.append(int(img[19:-5]))
    dict_images = {"ImageId": img_list, filename:cluster_list, "Numbering":imgno_list}
    model_labels = pd.DataFrame(dict_images).sort_values(by=["Numbering"])
    model_labels = model_labels[filename].tolist()
    
    return model_labels

def eval_table():

    ######################## Initialization ####################################
    general_path = os.getcwd()
    # imagenet_path = general_path + "\\util"
    imagenet_path = general_path + "/util"
    # DTclasslabels = pd.read_excel(imagenet_path + "\\LOC_val_solution.xlsx", sheet_name="Sheet1")
    DTclasslabels = pd.read_excel(imagenet_path + "/LOC_val_solution.xlsx", sheet_name="Sheet1")
    filenames = DTclasslabels["ImageId"].to_list()
    # pkl_files = os.listdir(general_path + "\\evaluation")
    pkl_files = os.listdir(general_path + "/evaluation")
    clustermodel_labels = DTclasslabels.drop(["Label","LabelNumber","Contents"], axis=1)

    ######################## Cluster Models ####################################
    df = []
    for file in pkl_files:
        # with open(general_path +"\\evaluation\\"+ file, 'rb') as f:
        with open(general_path +"/evaluation/"+ file, 'rb') as f:
            model_file = pickle.load(f)
            df.append({"Name": Path(file).stem, "Cluster size": len(model_file)}) 
            label = extract_labels(model_file, Path(file).stem)
            clustermodel_labels[Path(file).stem] = label
    df2 = {"Name": "VGG16", "Cluster size":1000}
    df.append(df2)
    df = pd.DataFrame(df)

    # modelfile = imagenet_path + '\\finalized_model.sav'
    modelfile = imagenet_path + '/finalized_model.sav'
    KMeansModel = pickle.load(open(modelfile, 'rb'))
    pred_list = pd.DataFrame(zip(filenames,KMeansModel.labels_), columns= ["ImageId","label"])
    clustermodel_labels["VGG16"] = pred_list["label"].tolist()

    ######################## Evaluations ####################################
    NMI_list, Rand_list, ARand_list= [], [], []
    for column in clustermodel_labels:
        if column == "ImageId":
            continue
        else:
            clusterlist = clustermodel_labels[column].tolist()
            NMI = normalized_mutual_info_score(DTclasslabels["LabelNumber"].tolist(), clusterlist)
            Rand = rand_score(DTclasslabels["LabelNumber"].tolist(),clusterlist)
            ARand = adjusted_rand_score(DTclasslabels["LabelNumber"].tolist(), clusterlist)
            NMI_list.append(str(round(NMI,3)))
            Rand_list.append(str(round(Rand,3)))
            ARand_list.append(str(round(ARand,5)))
        
    df["NMI"] = NMI_list
    df["Rand Score"] = Rand_list
    df["Adjusted Rand Score"] = ARand_list

    return df

def NMI_score(modelname, evaltable):

    NMI = evaltable.loc[evaltable["Name"] == modelname, "NMI"]
    NMI = str(float(NMI))

    return NMI

def ARand_score(modelname, evaltable):

    ARand = evaltable.loc[evaltable["Name"] == modelname, "Adjusted Rand Score"]
    ARand = str(float(ARand))

    return ARand
