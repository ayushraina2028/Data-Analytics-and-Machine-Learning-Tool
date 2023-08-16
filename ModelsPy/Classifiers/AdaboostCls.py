from flask import Blueprint, render_template, request, jsonify
AdaboostClassification = Blueprint('AdaboostClassification', __name__, template_folder='templates', static_folder='static')

# Adaboost Classification
def adaboost_classification(X_train,y_train,n_estimators,learning_rate,algorithm):
    
    from sklearn.ensemble import AdaBoostClassifier
    adaboost = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)
    adaboost.fit(X_train,y_train)
    
    return adaboost
    
@AdaboostClassification.route("/train_adaboost_classifier", methods = ["GET","POST"])
def train_adaboost_classifier():
    global adaboost_classifier
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    
    n_estimators = request.form.get("n_estimators")
    learning_rate = request.form.get("learning_rate")
    algorithm = request.form.get("algorithm")
    
    if not n_estimators:
        n_estimators=50
    else:
        n_estimators = int(n_estimators)
        
    if not learning_rate:
        learning_rate=1.0
    else:
        learning_rate = float(learning_rate)
        
    if not algorithm:
        algorithm="SAMME.R"
    
    adaboost_classifier=adaboost_classification(X_train,y_train,n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)
    return render_template("models/Boosting/Classifiers/AdaboostClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="adaboost_cls")

@AdaboostClassification.route("/test_adaboost_classifier", methods = ["GET","POST"])
def test_adaboost_classifier():
    
    from ModelsPy.Accuracies import check_accuracy
    
    score=check_accuracy(y_test,adaboost_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@AdaboostClassification.route("/predict_adaboost_cls", methods = ["GET","POST"])
def predict_adaboost_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = adaboost_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "AdaBoost Classifier", prediction=score[0])

