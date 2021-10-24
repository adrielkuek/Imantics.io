from flask import *
import glob, os, pprint, sys
from werkzeug.utils import secure_filename
from pathlib import Path
from models import *
from sklearn.neighbors import KNeighborsClassifier
from clipRetrieval import *
from DINOretrieval import *

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
    NNeighbor = int(request.form["clustersize"])
    
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
        filepath = os.path.join("static", "images","upload",secure_filename(f.filename))
        f.save(filepath)

    NNeighbor = int(request.form["clustersize"])
    imageResults_list = dino.kNN_retrieval(filepath, NNeighbor)
        
    return render_template('image.html',imageResults_list=imageResults_list, links=filelist, filepath=filepath)      

@app.route('/dino_50')
def dino_50():
    
    filelist = listofimages()
    dino_50cluster = Dino_50()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]    
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(50,dino_50cluster)

    return render_template('dino_50.html', links=filelist, dino_50=dino_50cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_100')
def dino_100():
    
    filelist = listofimages()
    dino_100cluster = Dino_100()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]    
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(100,dino_100cluster)

    return render_template('dino_100.html', links=filelist, dino_100=dino_100cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)


@app.route('/dino_150')
def dino_150():
    
    filelist = listofimages()
    dino_150cluster = Dino_150()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]    
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(150,dino_150cluster)

    return render_template('dino_150.html', links=filelist, dino_150=dino_150cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_200')
def dino_200():
    
    filelist = listofimages()
    dino_200cluster = Dino_200()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]    
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(200,dino_200cluster)

    return render_template('dino_200.html', links=filelist, dino_200=dino_200cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_250')
def dino_250():
    
    filelist = listofimages()
    dino_250cluster = Dino_250()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(250,dino_250cluster)

    return render_template('dino_250.html', links=filelist, dino_250=dino_250cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_300')
def dino_300():
    
    filelist = listofimages()
    dino_300cluster = Dino_300()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(300,dino_300cluster)

    return render_template('dino_300.html', links=filelist, dino_300=dino_300cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_350')
def dino_350():
    
    filelist = listofimages()
    dino_350cluster = Dino_350()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(350,dino_350cluster)

    return render_template('dino_350.html', links=filelist, dino_350=dino_350cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_400')
def dino_400():
    
    filelist = listofimages()
    dino_400cluster = Dino_400()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(400,dino_400cluster)

    return render_template('dino_400.html', links=filelist, dino_400=dino_400cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_450')
def dino_450():
    
    filelist = listofimages()
    dino_450cluster = Dino_450()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(450,dino_450cluster)

    return render_template('dino_450.html', links=filelist, dino_450=dino_450cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dino_500')
def dino_500():
    
    filelist = listofimages()
    dino_500cluster = Dino_500()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(500,dino_500cluster)

    return render_template('dino_500.html', links=filelist, dino_500=dino_500cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)


@app.route('/scan_200')
def scan_200():
    
    filelist = listofimages()
    scan_200cluster = Scan_200()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(200,scan_200cluster)

    return render_template('scan_200.html', links=filelist, scan_200=scan_200cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/vgg_200')
def vgg_200():
    
    filelist = listofimages()
    vgg_200cluster = Vgg_200()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(200,vgg_200cluster)

    return render_template('vgg_200.html', links=filelist, vgg_200=vgg_200cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_50')
def kmplus_dino_50():
    
    filelist = listofimages()
    kmplus_dino_50cluster = Kmplus_Dino_50()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(50,kmplus_dino_50cluster)

    return render_template('kmplus_dino_50.html', links=filelist, kmplus_dino_50=kmplus_dino_50cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_100')
def kmplus_dino_100():
    
    filelist = listofimages()
    kmplus_dino_100cluster = Kmplus_Dino_100()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(100,kmplus_dino_100cluster)

    return render_template('kmplus_dino_100.html', links=filelist, kmplus_dino_100=kmplus_dino_100cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_150')
def kmplus_dino_150():
    
    filelist = listofimages()
    kmplus_dino_150cluster = Kmplus_Dino_150()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(150,kmplus_dino_150cluster)

    return render_template('kmplus_dino_150.html', links=filelist, kmplus_dino_150=kmplus_dino_150cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_200')
def kmplus_dino_200():
    
    filelist = listofimages()
    kmplus_dino_200cluster = Kmplus_Dino_200()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(200,kmplus_dino_200cluster)

    return render_template('kmplus_dino_200.html', links=filelist, kmplus_dino_200=kmplus_dino_200cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_250')
def kmplus_dino_250():
    
    filelist = listofimages()
    kmplus_dino_250cluster = Kmplus_Dino_250()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(250,kmplus_dino_250cluster)

    return render_template('kmplus_dino_250.html', links=filelist, kmplus_dino_250=kmplus_dino_250cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_300')
def kmplus_dino_300():
    
    filelist = listofimages()
    kmplus_dino_300cluster = Kmplus_Dino_300()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(300,kmplus_dino_300cluster)

    return render_template('kmplus_dino_300.html', links=filelist, kmplus_dino_300=kmplus_dino_300cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_350')
def kmplus_dino_350():
    
    filelist = listofimages()
    kmplus_dino_350cluster = Kmplus_Dino_350()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(350,kmplus_dino_350cluster)

    return render_template('kmplus_dino_350.html', links=filelist, kmplus_dino_350=kmplus_dino_350cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_400')
def kmplus_dino_400():
    
    filelist = listofimages()
    kmplus_dino_400cluster = Kmplus_Dino_400()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(400,kmplus_dino_400cluster)

    return render_template('kmplus_dino_400.html', links=filelist, kmplus_dino_400=kmplus_dino_400cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/kmplus_dino_450')
def kmplus_dino_450():
    
    filelist = listofimages()
    kmplus_dino_450cluster = Kmplus_Dino_450()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(450,kmplus_dino_450cluster)

    return render_template('kmplus_dino_450.html', links=filelist, kmplus_dino_450=kmplus_dino_450cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_11')
def agglo_dino_11():
    
    filelist = listofimages()
    agglo_dino_11d = Agglo_Dino_11()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_11d),agglo_dino_11d)

    return render_template('agglo_dino_11.html', links=filelist, agglo_dino_11d=agglo_dino_11d, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_15')
def agglo_dino_15():
    
    filelist = listofimages()
    agglo_dino_15d = Agglo_Dino_15()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_15d),agglo_dino_15d)

    return render_template('agglo_dino_15.html', links=filelist, agglo_dino_15d=agglo_dino_15d, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_18')
def agglo_dino_18():
    
    filelist = listofimages()
    agglo_dino_18d = Agglo_Dino_18()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_18d),agglo_dino_18d)

    return render_template('agglo_dino_18.html', links=filelist, agglo_dino_18d=agglo_dino_18d, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_25')
def agglo_dino_25():
    
    filelist = listofimages()
    agglo_dino_25d = Agglo_Dino_25()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_25d),agglo_dino_25d)

    return render_template('agglo_dino_25.html', links=filelist, agglo_dino_25d=agglo_dino_25d, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_30')
def agglo_dino_30():
    
    filelist = listofimages()
    agglo_dino_30d = Agglo_Dino_30()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_30d),agglo_dino_30d)

    return render_template('agglo_dino_30.html', links=filelist, agglo_dino_30d=agglo_dino_30d, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dbscan_dino_eps8_m2')
def dbscan_dino_eps8_m2():
    
    filelist = listofimages()
    dbscan_dino_eps8_m2c = DBSCAN_Dino_EP8_M2()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(dbscan_dino_eps8_m2c),dbscan_dino_eps8_m2c)

    return render_template('dbscan_dino_eps8_m2.html', links=filelist, dbscan_dino_eps8_m2c=dbscan_dino_eps8_m2c, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dbscan_dino_eps8_m3')
def dbscan_dino_eps8_m3():
    
    filelist = listofimages()
    dbscan_dino_eps8_m3c = DBSCAN_Dino_EP8_M3()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(dbscan_dino_eps8_m3c),dbscan_dino_eps8_m3c)

    return render_template('dbscan_dino_eps8_m3.html', links=filelist, dbscan_dino_eps8_m3c=dbscan_dino_eps8_m3c, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/dbscan_dino_eps9_m2')
def dbscan_dino_eps9_m2():
    
    filelist = listofimages()
    dbscan_dino_eps9_m2c = DBSCAN_Dino_EP9_M2()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(dbscan_dino_eps9_m2c),dbscan_dino_eps9_m2c)

    return render_template('dbscan_dino_eps9_m2.html', links=filelist, dbscan_dino_eps9_m2c=dbscan_dino_eps9_m2c, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/optics_dino_m3')
def optics_dino_m3():
    
    filelist = listofimages()
    optics_dino_m3c = Optics_Dino_M3()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(optics_dino_m3c),optics_dino_m3c)

    return render_template('optics_dino_m3.html', links=filelist, optics_dino_m3c=optics_dino_m3c, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/optics_dino_m4')
def optics_dino_m4():
    
    filelist = listofimages()
    optics_dino_m4c = Optics_Dino_M4()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(optics_dino_m4c),optics_dino_m4c)

    return render_template('optics_dino_m4.html', links=filelist, optics_dino_m4c=optics_dino_m4c, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/optics_dino_m5')
def optics_dino_m5():
    
    filelist = listofimages()
    optics_dino_m5c = Optics_Dino_M5()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(optics_dino_m5c),optics_dino_m5c)

    return render_template('optics_dino_m5.html', links=filelist, optics_dino_m5c=optics_dino_m5c, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_ed12')
def agglo_dino_ed12():
    
    filelist = listofimages()
    agglo_dino_ed12c = Agglo_Dino_Ed12()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_ed12c),agglo_dino_ed12c)

    return render_template('agglo_dino_ed12.html', links=filelist, agglo_dino_ed12c=agglo_dino_ed12c, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_ed15')
def agglo_dino_ed15():
    
    filelist = listofimages()
    agglo_dino_ed15c = Agglo_Dino_Ed15()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_ed15c),agglo_dino_ed15c)

    return render_template('agglo_dino_ed15.html', links=filelist, agglo_dino_ed15c=agglo_dino_ed15c, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

@app.route('/agglo_dino_ed20')
def agglo_dino_ed20():
    
    filelist = listofimages()
    agglo_dino_ed20c = Agglo_Dino_Ed20()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(len(agglo_dino_ed20c),agglo_dino_ed20c)

    return render_template('agglo_dino_ed20.html', links=filelist, agglo_dino_ed20c=agglo_dino_ed20c, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
