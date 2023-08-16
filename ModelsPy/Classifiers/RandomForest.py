from flask import Blueprint, render_template, request, jsonify
RandomForestClassification = Blueprint('RandomForestClassification', __name__, template_folder='templates', static_folder='static')

# Random Forest Classification
def random_forest_classification(X_train,y_train,n_estimators,max_depth,max_features,criterion,bootstrap,oob_score):
    
    if max_depth == "None":
        max_depth = None 
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score)
    forest.fit(X_train,y_train)
    
    return forest

@RandomForestClassification.route("/train_random_forest_classifier", methods = ["GET","POST"])
def train_random_forest_classifier():
    global random_forest_classifier
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    
    n_estimators = request.form.get("n_estimators")
    max_depth = request.form.get("max_depth")
    max_features = request.form.get("max_features")
    criterion = request.form.get("criterion")
    bootstrap = request.form.get("bootstrap")
    oob_score = request.form.get("oob")
    
    
    if not n_estimators:
        n_estimators=100
    else:
        n_estimators = int(n_estimators)
        
    if not max_depth:
        max_depth=None
    else: 
        max_depth = int(max_depth)
        
    if not max_features:
        max_features="sqrt"
    elif max_features == "log2":
        max_features = "log2"
    elif max_features == "None":
        max_features = None
    else:
        max_features = float(max_features)
        
    if not criterion:
        criterion="gini"
    
    if not bootstrap:
        bootstrap=True
    else:
        bootstrap=False
        
    if not oob_score:
        oob_score=False
    else:
        oob_score=True
        
    random_forest_classifier=random_forest_classification(X_train,y_train,n_estimators=n_estimators, max_depth=max_depth,max_features=max_features, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score)
    return render_template("models/RandomForest/RandomForestClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="random_forest_cls")

@RandomForestClassification.route("/test_random_forest_classifier", methods = ["GET","POST"])
def test_random_forest_classifier():
    
    from ModelsPy.Accuracies import check_accuracy
    
    score=check_accuracy(y_test,random_forest_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@RandomForestClassification.route("/predict_random_forest_cls", methods = ["GET","POST"])
def predict_random_forest_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = random_forest_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Random Forest Classifier", prediction=score[0])
