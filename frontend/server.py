from flask import Flask
from flask import request, Response, render_template, flash, redirect, Blueprint, g, json, jsonify, session, url_for
import os
import zipfile

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
        files = request.files['file']
        filename = files.filename
        updir = os.path.join(basedir, 'upload/')
        zip_loc = os.path.join(updir, filename)
        files.save(zip_loc)

        with zipfile.ZipFile(zip_loc,"r") as zip_ref:
            zip_ref.extractall("user_train_data")
        os.remove(zip_loc)

        return jsonify(message = 'file uploaded successfully'), 200

    #req_json = request.get_json(silent=True) or request.form
    print("Acct_creation_Start!")
    #print(req_json)

    return jsonify(message = 'file uploaded successfully'), 200



if __name__ == "__main__":
    app.run(host='0.0.0.0',port= 1234)