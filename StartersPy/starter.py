from flask import Blueprint, render_template, request, redirect, url_for
starter = Blueprint('starter', __name__, template_folder='templates', static_folder='static')
import pandas as pd

@starter.route('/')
def home():
    return render_template("index.html")

@starter.route("/introduction")
def introduction():
    return render_template("Starters/intro.html")

#Upload Page
@starter.route("/upload")
def upload():
    return render_template("Starters/upload copy.html")

#Receiving the dataset here
@starter.route("/upload2", methods=['POST'])
def upload_dataset():
    global df
    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = pd.read_csv(file)
        
    return redirect(url_for("phase1.main_page"))