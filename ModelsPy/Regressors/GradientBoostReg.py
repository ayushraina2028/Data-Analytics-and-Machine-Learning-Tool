from flask import Blueprint, render_template, request, jsonify
GradientBoostRegression = Blueprint('GradientBoostRegression', __name__, template_folder='templates', static_folder='static')

# Gradient Boost Regressor
def gradient_boost_regression(X_train,y_train,n_estimators, learning_rate, loss, criterion, max_depth, max_features):
    
    from sklearn.ensemble import GradientBoostingRegressor
    gradient_boost = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, criterion=criterion, max_depth=max_depth, max_features=max_features)
    gradient_boost.fit(X_train,y_train)
    
    return gradient_boost

@GradientBoostRegression.route("/train_gradient_boost_regressor", methods = ["GET","POST"])
def train_gradient_boost_regressor():
    global gradient_boost_regressor
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    n_estimators = request.form.get("n_estimators")
    learning_rate = request.form.get("learning_rate")
    loss = request.form.get("loss")
    criterion = request.form.get("criterion")
    max_depth = request.form.get("max_depth")
    max_features = request.form.get("max_features")
    
    
    if not n_estimators:
        n_estimators=100
    else:
        n_estimators = int(n_estimators)
        
    if not learning_rate:
        learning_rate=0.1
    else:
        learning_rate = float(learning_rate)
        
    if not loss:
        loss="squared_error"
        
    if not criterion:
        criterion="friedman_mse"
        
    if not max_depth:
        max_depth=3
    else:
        max_depth = int(max_depth)
        
    if not max_features:
        max_features=None
    elif max_features == "log2":
        max_features = "log2"
    elif max_features == "None":
        max_features = None
    elif max_features == "sqrt":
        max_features = "sqrt"
    else:
        max_features = float(max_features)
        
    gradient_boost_regressor = gradient_boost_regression(X_train,y_train,n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, criterion=criterion, max_depth=max_depth, max_features=max_features)
    return render_template("models/Boosting/Regressors/GradientBoostRegressor.html",
                           training=X_train.shape, target=X_test.shape,train_status="Model is trained Successfully",
                           columns=training,model="gradient_boost_reg")

@GradientBoostRegression.route("/test_gradient_boost_regressor", methods = ["GET","POST"])
def test_gradient_boost_regressor():
    
    from ModelsPy.Accuracies import check_r2_score
    
    score=check_r2_score(y_test,gradient_boost_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@GradientBoostRegression.route("/predict_gradient_boost_reg", methods = ["GET","POST"])
def predict_gradient_boosting_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = gradient_boost_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Gradient Boosting Regression", prediction=score[0])
