from flask import Blueprint, render_template, request, jsonify
AdaboostRegression = Blueprint('AdaboostRegression', __name__, template_folder='templates', static_folder='static')

# Adaboost Regression
def adaboost_regression(X_train,y_train,n_estimators,learning_rate,loss):
    
    from sklearn.ensemble import AdaBoostRegressor
    adaboost = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss)
    adaboost.fit(X_train,y_train)
    
    return adaboost

@AdaboostRegression.route("/train_adaboost_regressor", methods = ["GET","POST"])
def train_adaboost_regressor():
    global adaboost_regressor
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    
    n_estimators = request.form.get("n_estimators")
    learning_rate = request.form.get("learning_rate")
    loss = request.form.get("loss")
    
    
    if not n_estimators:
        n_estimators=50
    else:
        n_estimators = int(n_estimators)
        
    if not learning_rate:
        learning_rate=1.0
    else:
        learning_rate = float(learning_rate)
        
    if not loss:
        loss="linear"
        
    
    
    adaboost_regressor = adaboost_regression(X_train,y_train,n_estimators=n_estimators, learning_rate=learning_rate, loss=loss)
    return render_template("models/Boosting/Regressors/AdaboostRegressor.html",
                           training=X_train.shape, target=X_test.shape,train_status="Model is trained Successfully",
                           columns=training,model="adaboost_reg")

@AdaboostRegression.route("/test_adaboost_regressor", methods = ["GET","POST"])
def test_adaboost_regressor():
    
    from ModelsPy.Accuracies import check_r2_score
    
    score=check_r2_score(y_test,adaboost_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@AdaboostRegression.route("/predict_adaboost_reg", methods = ["GET","POST"])
def predict_adaboost_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = adaboost_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "AdaBoost Regression", prediction=score[0])
