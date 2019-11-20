from flask import Flask
from flask import request, Response, render_template, flash, redirect, Blueprint, g, json, jsonify, session, url_for
import os
import zipfile
import smtplib
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText
import base64

from freeman import *
device = "cpu"

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def notify_completion(email, name):
    msg = MIMEMultipart()
    msg['From'] = 'doitpaws@gmail.com'
    msg['To'] = email
    msg['Subject'] = 'Account Creation on Project Freeman!'
    message = name + ', your account has been created on Project Freeman and is now ready for testing'
    msg.attach(MIMEText(message))

    mailserver = smtplib.SMTP('smtp.gmail.com',587)
    # identify ourselves to smtp gmail client
    mailserver.ehlo()
    # secure our email with tls encryption
    mailserver.starttls()
    # re-identify ourselves as an encrypted connection
    mailserver.ehlo()
    mailserver.login('doitpaws@gmail.com', 'temp_passw0rd_dumb@ass')

    mailserver.sendmail('doitpaws@gmail.com',email,msg.as_string())
    #mailserver.sendmail('me@gmail.com','you@gmail.com',msg.as_string())

    mailserver.quit()


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

        if not os.path.exists("user_account_models/"+account_name ):
            os.makedirs("user_account_models/"+account_name )
        else: 
            print("Account Already Exists!!!")
            return jsonify(message = 'Account Already Exists'), 200

        
        updir = os.path.join(basedir, 'upload/')
        zip_loc = os.path.join(updir, account_name)
        files.save(zip_loc)

        with zipfile.ZipFile(zip_loc,"r") as zip_ref:
            zip_ref.extractall("user_train_data")

        os.rename("user_train_data/"+ filename[0:-4] , "user_train_data/"+ account_name)

        os.remove(zip_loc)

        project = []
        project.append(instantiate([]))
        global most_recent_name
        most_recent_name = account_name

        #Always save project at end of run
        save_project("user_account_models/"+account_name, account_name ,project[0][1],project[0][0])
        
        print(" \nProgam Finished. \nExiting... Yeet")

        return jsonify(message = 'EZ 1'),200

    return jsonify(message = 'Nothing Happened'), 200



#Utils
@app.route("/acct_creation_v2", methods=['GET','POST'])
def acct_creation_2():
    if request.method == 'POST':
        req_json = request.get_json(silent=True) or request.form
        processed_json = req_json.to_dict()
        account_name = processed_json["name"]
        selfies =  json.loads(processed_json["images"])

        if not os.path.exists("user_train_data/"+account_name ):
            os.makedirs("user_train_data/"+account_name )
            os.makedirs("user_train_data/"+account_name+"/"+account_name)
            os.makedirs("user_account_models/"+account_name )
        else: 
            print("Account Already Exists!!!")
            return jsonify(message = 'Account Already Exists'), 200

        count = 0 
        for selfie in selfies:

            selfie = selfie.replace("data:image/jpeg;base64,","")
            print(selfie)
            imgdata = base64.b64decode(selfie)
        
            with open("user_train_data/"+account_name+"/"+account_name+"/"+account_name+str(count)+".jpeg", 'wb') as f:
                f.write(imgdata)

            count += 1

        project = []
        project.append(instantiate([]))
        global most_recent_name
        most_recent_name = account_name

        #Always save project at end of run
        save_project("user_account_models/"+account_name, account_name ,project[0][1],project[0][0])
        
        print(" \nProgam Finished. \nExiting... Yeet")

        return jsonify(message = 'EZ 1'),200

    return jsonify(message = 'Nothing Happened'), 200


@app.route("/train", methods=['GET','POST'])
def train():

    print("Made it!")
    print(most_recent_name)
    #load proj
    project = []
    project_trained= []

    req_json = request.get_json(silent=True) or request.form
    processed_json = req_json.to_dict()
    print(processed_json)
    name = processed_json["name"]
    email = processed_json["email"]


    project.append(load_project("user_account_models/"+name, name))

    
    #train 
    configurations = {
        "num_epochs": int(processed_json["epoch"]),
        "beta": float(processed_json["beta"]),
        "learning_rate": float(processed_json["learning_rate"]),
        "pos_source":"user_train_data/"+most_recent_name,
        "neg_source":"data/stock_neg_color"
    }


    project_trained.append( train_project_plus_negatives(project[0][0],project[0][1], configurations))

    # then save
    save_project("user_account_models/"+name, name ,project_trained[0][1],project_trained[0][0])
        
    #email noify
    print("stub notify")
    if(email):
        notify_completion(email,name )
    

    return jsonify(message = 'EZ 2'), 200



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

        
        validate(project[0][1],project[0][0],"user_account_models/"+account_name+"/"+account_name+"_verify")


        #os.remove("user_account_models/"+account_name+"/"+account_name+"_verify")
        print(" \nProgam Finished 2. \nExiting 2... Yeet")
        

        return jsonify(message = 'file verified successfully'), 200

    #req_json = request.get_json(silent=True) or request.form
    print("Acct Verify DOne!")
    #print(req_json)

    return jsonify(message = 'file verified successfully'), 200

import cv2

#Utils
@app.route("/acct_test_v2", methods=['GET','POST'])
def acct_test_v2():
    if request.method == 'POST':
        req_json = request.get_json(silent=True) or request.form
        processed_json = req_json.to_dict()

        account_name = processed_json["name"]
        print(account_name)

        project = []
        
        if os.path.exists("user_account_models/"+account_name):
            project.append(load_project("user_account_models/"+account_name, account_name))
        else:
            print("No Project Found")
            return jsonify(message = 'No Acct'), 200


        pic = processed_json["image"]
        pic = pic.replace("data:image/jpeg;base64,","")

        imgdata = base64.b64decode(pic)
        
        with open("user_account_models/"+account_name+"/"+account_name+"_verify", 'wb') as f:
            f.write(imgdata)

        
        validate(project[0][1],project[0][0],"user_account_models/"+account_name+"/"+account_name+"_verify")



        return jsonify(message = 'file_v2 verified successfully'), 200

    return jsonify(message = 'file_v2 verified successfully'), 200




if __name__ == "__main__":
    most_recent_name = ""
    app.run(host='0.0.0.0',port= 1234)