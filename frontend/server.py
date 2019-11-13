from flask import Flask
from flask import request, Response, render_template, flash, redirect, Blueprint, g, json, jsonify, session, url_for

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(host='0.0.0.0',port= 1234)