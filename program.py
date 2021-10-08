from flask import *
import glob, os, pprint
from werkzeug.utils import secure_filename
from pathlib import Path
from CLIP import *

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "images")

@app.route('/')
def home_form():

    ########### Multiple files #################    
##    folderlist = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"],"Cluster *"))

    
    folderlist = glob.glob(os.path.join(os.getcwd(),"static", "images") + "/*/")
    foldergroup = [Path(folder).stem for folder in folderlist]
    filelist = glob.glob(os.path.join(app.config["UPLOAD_FOLDER"],"*"))
    groupings = CLIP()
    
##    folderdict = {Path(folder).stem:[] for folder in folderlist}
##    for foldername in folderdict.keys():
##        for file in filelist:    
##            if foldername in file:
##                folderdict[foldername].append(file)

    return render_template('query_by_cluster_v2.html', links=filelist, foldergroup=foldergroup, folderdict=groupings)

    ########### Single file #####################
    #fullname = os.path.join(app.config['UPLOAD_FOLDER'], '00001.jpg')
    #return render_template('query.html', user_image=fullname)


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
