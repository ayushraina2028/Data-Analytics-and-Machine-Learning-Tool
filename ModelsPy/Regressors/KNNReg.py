from flask import Blueprint, render_template, request, jsonify
KNNRegression = Blueprint('KNNRegression', __name__, template_folder='templates', static_folder='static')

# KNN Regressor
def knn_regression(X_train,y_train,n_neighbors,weights,algorithm,leaf_size,p):
    
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
    knn.fit(X_train,y_train)
    
    return knn

@KNNRegression.route("/train_knn_regressor", methods = ["GET","POST"])
def train_knn_regressor():
    global knn_regressor
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    
    n_neighbors = request.form.get("n_neighbors")
    weights = request.form.get("weights")
    algorithm = request.form.get("algorithm")
    leaf_size = request.form.get("leaf_size")
    p = request.form.get("p")
    
    if not n_neighbors:
        n_neighbors=5
    else:
        n_neighbors = int(n_neighbors)
        
    if not weights:
        weights="uniform"
    
        
    if not algorithm:
        algorithm="auto"
        
    if not leaf_size:
        leaf_size=30
    else:
        leaf_size = int(leaf_size)
        
    if not p:
        p=2
    else:
        p = int(p)
        
    knn_regressor=knn_regression(X_train,y_train,n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
    return render_template("models/KNearestNeighbours/KNNRegressor.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model = "knn_reg")
    
@KNNRegression.route("/test_knn_regressor", methods = ["GET","POST"])
def test_knn_regressor():
    
        from ModelsPy.Accuracies import check_r2_score
        
        score=check_r2_score(y_test,knn_regressor.predict(X_test))
        score=score*100
        return jsonify({"score":score})

@KNNRegression.route("/predict_knn_reg", methods = ["GET","POST"])
def predict_knn_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = knn_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "K Nearest Neighbors Regression", prediction=score[0])


