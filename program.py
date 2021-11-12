from flask import *
import glob, os, pprint, sys
from werkzeug.utils import secure_filename
from pathlib import Path
from models import *
from sklearn.neighbors import KNeighborsClassifier
from clipRetrieval import *
from DINOretrieval import *
from model_evaluation import *
import natsort

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "images")

######## Initialization of CLIP Retrieval Functions ########################
clipmodel_dir = os.getcwd() + "/models/ViT-B-32.pt"
clipinput_filename_pickle =  os.getcwd() + "/feature_tensors/clip_retrieval/image_filenames_OurDataset.pkl"
clipdataset_tensor = os.getcwd()+ "/feature_tensors/clip_retrieval/image_embeddings_OurDataset.pt"

dinomodel_dir = os.getcwd() + "/models/dino_deitsmall8_pretrain.pth"
dinoinput_filename_pickle = os.getcwd() + "/feature_tensors/dino_retrieval/Ourdataset_loader.pth"
dinodataset_tensor = os.getcwd() + "/feature_tensors/dino_retrieval/Ourdataset_features.pth"

kNNstep = 5
kNNmax = 50
kNNstart = 5

clip = CLIP(clipmodel_dir, clipdataset_tensor, clipinput_filename_pickle)
dino = DINO(dinomodel_dir, dinoinput_filename_pickle, dinodataset_tensor)

evaltable = eval_table()

######## Defined functions ##################################################
def listofimages():
    folderlist = glob.glob(os.path.join(os.getcwd(),"static", "images") + "/*/")
    foldergroup = [Path(folder).stem for folder in folderlist]
    filelist = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"],"*"))

    return filelist

############# Website Configuration #########################################
@app.route('/')
def home_form():

    return render_template('home.html')

@app.route("/text", methods=['POST'])
def text():
    imageResults_list = []
    filelist = listofimages()    
    querytext = request.form["text"]

    try:
        NNeighbor = int(request.form["clustersize"])
    except:
        NNeighbor = 20
    
    imageResults_list = clip.kNN_retrieval(NNeighbor, querytext, is_text=True)
    return render_template('text.html', imageResults_list=imageResults_list, links=filelist, querytext=querytext)

@app.route('/image', methods = ['POST','GET'])
def upload():
    imageResults_list = []
    filelist = listofimages()
    
    if request.method == 'POST':
        imagelist = glob.glob(os.path.join(os.getcwd(),"static", "images","upload","*"))
        for image in imagelist:
            os.remove(image)

        f = request.files['File']
        print(f)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"],"upload",secure_filename(f.filename))
        f.save(filepath)

    try:
        NNeighbor = int(request.form["clustersize"])
    except:
        NNeighbor = 20
        
    imageResults_list = dino.kNN_retrieval(filepath, NNeighbor)
        
    return render_template('image.html',imageResults_list=imageResults_list, filepath=filepath, links=filelist)      

@app.route('/dino_50')
def dino_50():
    
    filelist = listofimages()
    dino_50cluster = Dino_50()
    NMI = NMI_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]    
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(50,dino_50cluster)

    return render_template('dino_50.html', links=filelist, dino_50=dino_50cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_100')
def dino_100():
    
    filelist = listofimages()
    dino_100cluster = Dino_100()
    NMI = NMI_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]    
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(100,dino_100cluster)

    return render_template('dino_100.html', links=filelist, dino_100=dino_100cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)


@app.route('/dino_150')
def dino_150():
    
    filelist = listofimages()
    dino_150cluster = Dino_150()
    NMI = NMI_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]    
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(150,dino_150cluster)

    return render_template('dino_150.html', links=filelist, dino_150=dino_150cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_200')
def dino_200():
    
    filelist = listofimages()
    dino_200cluster = Dino_200()
    NMI = NMI_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]    
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(200,dino_200cluster)

    return render_template('dino_200.html', links=filelist, dino_200=dino_200cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_250')
def dino_250():
    
    filelist = listofimages()
    dino_250cluster = Dino_250()
    NMI = NMI_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(250,dino_250cluster)

    return render_template('dino_250.html', links=filelist, dino_250=dino_250cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_300')
def dino_300():
    
    filelist = listofimages()
    dino_300cluster = Dino_300()
    NMI = NMI_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(300,dino_300cluster)

    return render_template('dino_300.html', links=filelist, dino_300=dino_300cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_350')
def dino_350():
    
    filelist = listofimages()
    dino_350cluster = Dino_350()
    NMI = NMI_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(350,dino_350cluster)

    return render_template('dino_350.html', links=filelist, dino_350=dino_350cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_400')
def dino_400():
    
    filelist = listofimages()
    dino_400cluster = Dino_400()
    NMI = NMI_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(400,dino_400cluster)

    return render_template('dino_400.html', links=filelist, dino_400=dino_400cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_450')
def dino_450():
    
    filelist = listofimages()
    dino_450cluster = Dino_450()
    NMI = NMI_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(450,dino_450cluster)

    return render_template('dino_450.html', links=filelist, dino_450=dino_450cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_500')
def dino_500():
    
    filelist = listofimages()
    dino_500cluster = Dino_500()
    NMI = NMI_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeans_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(500,dino_500cluster)

    return render_template('dino_500.html', links=filelist, dino_500=dino_500cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)


@app.route('/scan_200')
def scan_200():
    
    filelist = listofimages()
    scan_200cluster = Scan_200()
    NMI = NMI_score("SCAN_1kclusters_imagenet",evaltable)
    ARand = ARand_score("SCAN_1kclusters_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(200,scan_200cluster)

    return render_template('scan_200.html', links=filelist, scan_200=scan_200cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/vgg_200')
def vgg_200():
    
    filelist = listofimages()
    vgg_200cluster = Vgg_200()
    NMI = NMI_score("VGG16",evaltable)
    ARand = ARand_score("VGG16",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(200,vgg_200cluster)

    return render_template('vgg_200.html', links=filelist, vgg_200=vgg_200cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_50')
def kmplus_dino_50():
    
    filelist = listofimages()
    kmplus_dino_50cluster = Kmplus_Dino_50()
    NMI = NMI_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(50,kmplus_dino_50cluster)

    return render_template('kmplus_dino_50.html', links=filelist, kmplus_dino_50=kmplus_dino_50cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_100')
def kmplus_dino_100():
    
    filelist = listofimages()
    kmplus_dino_100cluster = Kmplus_Dino_100()
    NMI = NMI_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(100,kmplus_dino_100cluster)

    return render_template('kmplus_dino_100.html', links=filelist, kmplus_dino_100=kmplus_dino_100cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_150')
def kmplus_dino_150():
    
    filelist = listofimages()
    kmplus_dino_150cluster = Kmplus_Dino_150()
    NMI = NMI_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(150,kmplus_dino_150cluster)

    return render_template('kmplus_dino_150.html', links=filelist, kmplus_dino_150=kmplus_dino_150cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_200')
def kmplus_dino_200():
    
    filelist = listofimages()
    kmplus_dino_200cluster = Kmplus_Dino_200()
    NMI = NMI_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(200,kmplus_dino_200cluster)

    return render_template('kmplus_dino_200.html', links=filelist, kmplus_dino_200=kmplus_dino_200cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_250')
def kmplus_dino_250():
    
    filelist = listofimages()
    kmplus_dino_250cluster = Kmplus_Dino_250()
    NMI = NMI_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(250,kmplus_dino_250cluster)

    return render_template('kmplus_dino_250.html', links=filelist, kmplus_dino_250=kmplus_dino_250cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_300')
def kmplus_dino_300():
    
    filelist = listofimages()
    kmplus_dino_300cluster = Kmplus_Dino_300()
    NMI = NMI_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(300,kmplus_dino_300cluster)

    return render_template('kmplus_dino_300.html', links=filelist, kmplus_dino_300=kmplus_dino_300cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_350')
def kmplus_dino_350():
    
    filelist = listofimages()
    kmplus_dino_350cluster = Kmplus_Dino_350()
    NMI = NMI_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(350,kmplus_dino_350cluster)

    return render_template('kmplus_dino_350.html', links=filelist, kmplus_dino_350=kmplus_dino_350cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_400')
def kmplus_dino_400():
    
    filelist = listofimages()
    kmplus_dino_400cluster = Kmplus_Dino_400()
    NMI = NMI_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(400,kmplus_dino_400cluster)

    return render_template('kmplus_dino_400.html', links=filelist, kmplus_dino_400=kmplus_dino_400cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_450')
def kmplus_dino_450():
    
    filelist = listofimages()
    kmplus_dino_450cluster = Kmplus_Dino_450()
    NMI = NMI_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    ARand = ARand_score("DINO_kmeansplus_1kcluster_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(450,kmplus_dino_450cluster)

    return render_template('kmplus_dino_450.html', links=filelist, kmplus_dino_450=kmplus_dino_450cluster, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dbscan_dino_eps8_m2')
def dbscan_dino_eps8_m2():
    
    filelist = listofimages()
    dbscan_dino_eps8_m2c = DBSCAN_Dino_EP8_M2()
    NMI = NMI_score("DINO_dbscan_eps_08_minsamples2_imagenet",evaltable)
    ARand = ARand_score("DINO_dbscan_eps_08_minsamples2_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(dbscan_dino_eps8_m2c),dbscan_dino_eps8_m2c)

    return render_template('dbscan_dino_eps8_m2.html', links=filelist, dbscan_dino_eps8_m2c=dbscan_dino_eps8_m2c,NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dbscan_dino_eps8_m3')
def dbscan_dino_eps8_m3():
    
    filelist = listofimages()
    dbscan_dino_eps8_m3c = DBSCAN_Dino_EP8_M3()
    NMI = NMI_score("DINO_dbscan_eps_08_minsamples3_imagenet",evaltable)
    ARand = ARand_score("DINO_dbscan_eps_08_minsamples3_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(dbscan_dino_eps8_m3c),dbscan_dino_eps8_m3c)

    return render_template('dbscan_dino_eps8_m3.html', links=filelist, dbscan_dino_eps8_m3c=dbscan_dino_eps8_m3c, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dbscan_dino_eps9_m2')
def dbscan_dino_eps9_m2():
    
    filelist = listofimages()
    dbscan_dino_eps9_m2c = DBSCAN_Dino_EP9_M2()
    NMI = NMI_score("DINO_dbscan_eps_09_minsamples2_imagenet",evaltable)
    ARand = ARand_score("DINO_dbscan_eps_09_minsamples2_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(dbscan_dino_eps9_m2c),dbscan_dino_eps9_m2c)

    return render_template('dbscan_dino_eps9_m2.html', links=filelist, dbscan_dino_eps9_m2c=dbscan_dino_eps9_m2c, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/optics_dino_m3')
def optics_dino_m3():
    
    filelist = listofimages()
    optics_dino_m3c = Optics_Dino_M3()
    NMI = NMI_score("DINO_optics_minsample_3_imagenet",evaltable)
    ARand = ARand_score("DINO_optics_minsample_3_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(optics_dino_m3c),optics_dino_m3c)

    return render_template('optics_dino_m3.html', links=filelist, optics_dino_m3c=optics_dino_m3c, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/optics_dino_m4')
def optics_dino_m4():
    
    filelist = listofimages()
    optics_dino_m4c = Optics_Dino_M4()
    NMI = NMI_score("DINO_optics_minsample_4_imagenet",evaltable)
    ARand = ARand_score("DINO_optics_minsample_4_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(optics_dino_m4c),optics_dino_m4c)

    return render_template('optics_dino_m4.html', links=filelist, optics_dino_m4c=optics_dino_m4c, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/optics_dino_m5')
def optics_dino_m5():
    
    filelist = listofimages()
    optics_dino_m5c = Optics_Dino_M5()
    NMI = NMI_score("DINO_optics_minsample_5_imagenet",evaltable)
    ARand = ARand_score("DINO_optics_minsample_5_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(optics_dino_m5c),optics_dino_m5c)

    return render_template('optics_dino_m5.html', links=filelist, optics_dino_m5c=optics_dino_m5c, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_ed12')
def agglo_dino_ed12():
    
    filelist = listofimages()
    agglo_dino_ed12c = Agglo_Dino_Ed12()
    NMI = NMI_score("DINO_agglo_eucldist1_2_imagenet",evaltable)
    ARand = ARand_score("DINO_agglo_eucldist1_2_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_ed12c),agglo_dino_ed12c)

    return render_template('agglo_dino_ed12.html', links=filelist, agglo_dino_ed12c=agglo_dino_ed12c, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_ed15')
def agglo_dino_ed15():
    
    filelist = listofimages()
    agglo_dino_ed15c = Agglo_Dino_Ed15()
    NMI = NMI_score("DINO_agglo_eucldist1_5_imagenet",evaltable)
    ARand = ARand_score("DINO_agglo_eucldist1_5_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_ed15c),agglo_dino_ed15c)

    return render_template('agglo_dino_ed15.html', links=filelist, agglo_dino_ed15c=agglo_dino_ed15c, NMI=NMI, ARand=ARand, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_ed20')
def agglo_dino_ed20():
    
    filelist = listofimages()
    agglo_dino_ed20c = Agglo_Dino_Ed20()
    NMI = NMI_score("DINO_agglo_eucldist2_0_imagenet",evaltable)
    ARand = ARand_score("DINO_agglo_eucldist2_0_imagenet",evaltable)
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_ed20c),agglo_dino_ed20c)

    return render_template('agglo_dino_ed20.html', links=filelist, agglo_dino_ed20c=agglo_dino_ed20c, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
