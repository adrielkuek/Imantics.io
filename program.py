from flask import *
import glob, os, pprint
from werkzeug.utils import secure_filename
from pathlib import Path
from models import *
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "images")

def listofimages():
    folderlist = glob.glob(os.path.join(os.getcwd(),"static", "images") + "/*/")
    foldergroup = [Path(folder).stem for folder in folderlist]
    filelist = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"],"*"))

    return filelist

@app.route('/')
def home_form():

    return render_template('home.html')

@app.route("/text", methods=['POST'])
def text():
    query = request.form["text"]   
    return render_template('text.html', query=query)

@app.route('/upload', methods = ['POST','GET'])
def upload():
    if request.method == 'POST':
        imagelist = glob.glob(os.path.join(os.getcwd(),"instance","*"))
        for image in imagelist:
            os.remove(image)

        f = request.files['File']
##        f.save(secure_filename(f.filename))
        filepath = os.path.join("instance",secure_filename(f.filename))
        f.save(filepath)
        return render_template('image.html')      

@app.route('/dino_200')
def dino_200():
    
    filelist = listofimages()
    dino_200cluster = Dino_200()

    return render_template('dino_200.html', links=filelist, dino_200=dino_200cluster)

@app.route('/dino_250')
def dino_250():
    
    filelist = listofimages()
    dino_250cluster = Dino_250()

    return render_template('dino_250.html', links=filelist, dino_250=dino_250cluster)

@app.route('/dino_300')
def dino_300():
    
    filelist = listofimages()
    dino_300cluster = Dino_300()

    return render_template('dino_300.html', links=filelist, dino_300=dino_300cluster)

@app.route('/dino_350')
def dino_350():
    
    filelist = listofimages()
    dino_350cluster = Dino_350()

    return render_template('dino_350.html', links=filelist, dino_350=dino_350cluster)

@app.route('/dino_400')
def dino_400():
    
    filelist = listofimages()
    dino_400cluster = Dino_400()

    return render_template('dino_400.html', links=filelist, dino_400=dino_400cluster)

@app.route('/dino_450')
def dino_450():
    
    filelist = listofimages()
    dino_450cluster = Dino_450()

    return render_template('dino_450.html', links=filelist, dino_450=dino_450cluster)

@app.route('/scan_200')
def scan_200():
    
    filelist = listofimages()
    scan_200cluster = Scan_200()

    return render_template('scan_200.html', links=filelist, scan_200=scan_200cluster)

##@app.route('/')
##def home():
##    return render_template('home.html')
##
##@app.route('/about/')
##def about():
##    return render_template('about.html')
##
##@app.route('/person/')
##def hello():
##    return jsonify({'name':'Jimit',
##                    'address':'India'})
##
##@app.route('/numbers/')
##def print_list():
##    return jsonify(list(range(5)))

##@app.route('/teapot/')
##def teapot():
##    return "Would you like some tea?", 418

##@app.before_request
##def before():
##    print("This is executed BEFORE each request.")
##    
##@app.route('/hello/')
##def hello():
##    return "Hello World!"

################################################
##def getParent(path, levels = 1):
##    common = path
##    for i in range(levels + 1):
##        
##        # Starting point
##        common = os.path.dirname(common)
## 
##    return os.path.relpath(path, common)
##
#####################################################
##    for folder in folderlist:
##        for file in filelist:
##            if getParent(folder,1) == (getParent(file,1)).split("\\")[0]:
##                folderdict[getParent(folder,1)].append(file)


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
