# Required Dependencies
from flask import Flask, send_file
import io
import warnings
warnings.filterwarnings("ignore")

# Importing Blue Print
from StartersPy.starter import starter
from StartersPy.Phase1  import phase1
from StartersPy.missvalue import missvalue
from StartersPy.Encoding import encoder
from StartersPy.TrainTestSplit import trainTest
from StartersPy.GraphTool import GraphTool
from ModelsPy.modelPage import modelPage
from ModelsPy.Regressors.LinearRegression import LinearRegression
from ModelsPy.Regressors.DecisionTreeReg import DecisionTreeRegression
from ModelsPy.Regressors.SupportVectorReg import SupportVectorRegression
from ModelsPy.Regressors.RandomForest import RandomForestRegression
from ModelsPy.Regressors.AdaboostReg import AdaboostRegression
from ModelsPy.Regressors.GradientBoostReg import GradientBoostRegression
from ModelsPy.Regressors.XGBoostReg import XGBoostRegression
from ModelsPy.Regressors.KNNReg import KNNRegression
from ModelsPy.Classifiers.LogisticRegression import LogisticRegression
from ModelsPy.Classifiers.DecisionTreeCls import DecisionTreeClassification
from ModelsPy.Classifiers.NaiveBayes import NaiveBayesClassification
from ModelsPy.Classifiers.SupportVectorCls import SupportVectorClassification
from ModelsPy.Classifiers.RandomForest import RandomForestClassification
from ModelsPy.Classifiers.AdaboostCls import AdaboostClassification
from ModelsPy.Classifiers.KNNCls import KNNClassification
from ModelsPy.Classifiers.GradientBoostCls import GradientBoostClassification


# Registering Blue Prints
app = Flask(__name__)
app.register_blueprint(starter)
app.register_blueprint(phase1)
app.register_blueprint(missvalue)
app.register_blueprint(encoder)
app.register_blueprint(trainTest)
app.register_blueprint(GraphTool)
app.register_blueprint(modelPage)
app.register_blueprint(LinearRegression)
app.register_blueprint(LogisticRegression)
app.register_blueprint(DecisionTreeRegression)
app.register_blueprint(SupportVectorRegression)
app.register_blueprint(RandomForestRegression)
app.register_blueprint(AdaboostRegression)
app.register_blueprint(GradientBoostRegression)
app.register_blueprint(XGBoostRegression)
app.register_blueprint(KNNRegression)
app.register_blueprint(DecisionTreeClassification)
app.register_blueprint(NaiveBayesClassification)
app.register_blueprint(SupportVectorClassification)
app.register_blueprint(RandomForestClassification)
app.register_blueprint(AdaboostClassification)
app.register_blueprint(KNNClassification)
app.register_blueprint(GradientBoostClassification)

# Route for Downloading dataset
@app.route("/download")
def download():
    
    global send,df
    from StartersPy.Encoding import df
    
    csv_file=io.StringIO()
    df.to_csv(csv_file,index=False)
    
    return send_file(
        io.BytesIO(csv_file.getvalue().encode()),
        as_attachment=True,
        download_name="Dataset.csv",
        mimetype='text/csv'
    )

if __name__=="__main__":
    app.run(host="0.0.0.0")

