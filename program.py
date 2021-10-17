from flask import *
import glob, os, pprint
from werkzeug.utils import secure_filename
from pathlib import Path
from models import *
from sklearn.neighbors import KNeighborsClassifier
from clipRetrieval import *

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "images")

######## Initialization of CLIP Retrieval Functions ########################
model_dir = "C:\\Users\\Chua Hz\\Desktop\\Project\\models\\ViT-B-32.pt"
input_filename_pickle = "C:\\Users\\Chua Hz\\Desktop\\Project\\feature_tensors\\CLIP\\image_filenames_OurDataset.pkl"
dataset_tensor = "C:\\Users\\Chua Hz\\Desktop\\Project\\feature_tensors\\CLIP\\image_embeddings_OurDataset.pt"

##model_dir = '/home/user/Adriel/MTech-IS/GradCert_PatternRecognition/PRS_PractiseModule/Imantics.io/ViT-B-32.pt'
##input_filename_pickle = '/home/user/Adriel/MTech-IS/GradCert_PatternRecognition/PRS_PractiseModule/Imantics.io/OurDataset_Tensors/image_filenames_OurDataset.pkl'
##dataset_tensor = '/home/user/Adriel/MTech-IS/GradCert_PatternRecognition/PRS_PractiseModule/Imantics.io/OurDataset_Tensors/image_embeddings_OurDataset.pt'

kNNstep = 5
kNNmax = 50
kNNstart = 5

clip = CLIP(model_dir, dataset_tensor, input_filename_pickle)

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
    NNeighbor = request.form["clustersize"]
    
    #imageResults_list = clip.kNN_retrieval(NNeighbor, querytext, is_text=True)
    return render_template('text.html', imageResults_list=imageResults_list, links=filelist)

@app.route('/image', methods = ['POST','GET'])
def upload():
    imageResults_list = []
    
    if request.method == 'POST':
        imagelist = glob.glob(os.path.join(os.getcwd(),"static", "images","upload","*"))
        for image in imagelist:
            os.remove(image)

        f = request.files['File']
        filepath = os.path.join("static", "images","upload",secure_filename(f.filename))
        f.save(filepath)

    queryImage = Image.open(filepath)
    #NNeighbor = request.form["clustersize"]

    #imageResults_list = clip.kNN_retrieval(NNeighbor, queryImage, is_text=False)
        
    return render_template('image.html',imageResults_list=imageResults_list, filepath=filepath)      

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

@app.route('/scan_200')
def scan_200():
    
    filelist = listofimages()
    scan_200cluster = Scan_200()
    listofN = [str(int) for int in list(range(kNNstart,kNNmax+1,kNNstep))]
    imgcount_zero, imgcount_nonzero, max_value, min_value, mean_value = stats(200,scan_200cluster)

    return render_template('scan_200.html', links=filelist, scan_200=scan_200cluster, listofN=listofN, imgcount_zero=imgcount_zero, imgcount_nonzero=imgcount_nonzero, max_value=max_value, min_value=min_value, mean_value=mean_value)

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
