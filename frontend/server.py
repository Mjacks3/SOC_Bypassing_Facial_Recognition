from flask import Flask
from flask import request, Response, render_template, flash, redirect, Blueprint, g, json, jsonify, session, url_for
import os
import zipfile

from freeman import *
device = "cpu"

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route("/hello")
def hello():
    return "Hello World!"

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('bout.html')

@app.route("/existing")
def existing():
    return render_template('existing.html')

@app.route("/new")
def new():
    return render_template('new.html')


#Utils
@app.route("/acct_creation", methods=['GET','POST'])
def acct_creation():
    if request.method == 'POST':
        account_name = list(request.files.to_dict().keys())[0]
        files = request.files[account_name]
        filename = files.filename

        """
        updir = os.path.join(basedir, 'upload/')
        zip_loc = os.path.join(updir, filename)
        files.save(zip_loc)

        with zipfile.ZipFile(zip_loc,"r") as zip_ref:
            zip_ref.extractall("user_train_data")
        os.remove(zip_loc)
        """
        project = []
        name = str(manualSeed)
        project.append(instantiate([]))


        if not os.path.exists("user_account_models/"+account_name ):
            os.makedirs("user_account_models/"+account_name )
        else: 
            print("Account Already Exits!!!")
            return jsonify(message = 'Account Already Exists'), 200


        #Always save project at end of run
        save_project("user_account_models/"+account_name, account_name ,project[0][1],project[0][0])
        

        print(" \nProgam Finished. \nExiting... Yeet")

        return jsonify(message = 'file uploaded successfully'), 200

    #req_json = request.get_json(silent=True) or request.form
    print("Acct_creation_Start!")
    #print(req_json)

    return jsonify(message = 'file uploaded successfully'), 200


#Utils
@app.route("/acct_test", methods=['GET','POST'])
def acct_test():
    if request.method == 'POST':
        account_name = list(request.files.to_dict().keys())[0]
        files = request.files[account_name]
        filename = files.filename
        print(account_name)

        project = []
        
        if os.path.exists("user_account_models/"+account_name):
            project.append(load_project("user_account_models/"+account_name, account_name))

        else:
            print("No Project Found")
            return jsonify(message = 'No Acct'), 200


        files.save("user_account_models/"+account_name+"/"+account_name+"_verify")

        
        validate(project[0][0],project[0][1],"user_account_models/"+account_name+"/"+account_name+"_verify")


        #os.remove("user_account_models/"+account_name+"/"+account_name+"_verify")
        print(" \nProgam Finished 2. \nExiting 2... Yeet")
        

        return jsonify(message = 'file verified successfully'), 200

    #req_json = request.get_json(silent=True) or request.form
    print("Acct Verify DOne!")
    #print(req_json)

    return jsonify(message = 'file verified successfully'), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0',port= 1234)