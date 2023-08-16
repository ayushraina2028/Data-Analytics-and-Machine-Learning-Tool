from flask import Blueprint, render_template, request, jsonify
RandomForestRegression = Blueprint('RandomForestRegression', __name__, template_folder='templates', static_folder='static')


# Random Forest Regression
def random_forest_regression(X_train,y_train,n_estimators,max_depth,max_features,criterion,bootstrap,oob_score):
    
    if max_depth == "None":
        max_depth = None 
    from sklearn.ensemble import RandomForestRegressor
    forest = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score)
    forest.fit(X_train,y_train)
    
    return forest

@RandomForestRegression.route("/train_random_forest_regressor", methods = ["GET","POST"])
def train_random_forest_regressor():
    global random_forest_regressor
    
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
        max_features=1
    elif max_features == "log2":
        max_features = "log2"
    elif max_features == "None":
        max_features = None
    elif max_features == "sqrt":
        max_features = "sqrt"
    else:
        max_features = float(max_features)
        
    if not criterion:
        criterion="squared_error"
    
    if not bootstrap:
        bootstrap=True
    else:
        bootstrap=False
        
    if not oob_score:
        oob_score=False
    else:
        oob_score=True
    
    random_forest_regressor = random_forest_regression(X_train,y_train,n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score)
    return render_template("models/RandomForest/RandomForestRegressor.html",
                           training=X_train.shape, target=X_test.shape,train_status="Model is trained Successfully",
                           columns=training,model="random_forest_reg")

@RandomForestRegression.route("/test_random_forest_regressor", methods = ["GET","POST"])
def test_random_forest_regressor():
    
    from ModelsPy.Accuracies import check_r2_score
    
    score=check_r2_score(y_test,random_forest_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@RandomForestRegression.route("/predict_random_forest_reg", methods = ["GET","POST"])
def predict_random_forest_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = random_forest_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Random Forest Regression", prediction=score[0])

