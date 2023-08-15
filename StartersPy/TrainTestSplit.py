from flask import Blueprint, render_template, request, redirect, url_for
trainTest = Blueprint('trainTest', __name__, template_folder='templates', static_folder='static')

@trainTest.route("/show_tts")
def tts():
    
    global df
    from StartersPy.Encoding import df
    
    return render_template("ML/tts.html",columns=df.columns.to_list())