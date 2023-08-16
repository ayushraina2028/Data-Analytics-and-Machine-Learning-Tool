from flask import Blueprint, render_template, request, jsonify
XGBoostRegression = Blueprint('XGBoostRegression', __name__, template_folder='templates', static_folder='static')

# Extreme Gradient booost
import xgboost as xbs

# XGBoost Regressor
def xgboost_regression(X_train,y_train):
    
    xgb_reg=xbs.XGBRegressor()
    xgb_reg.fit(X_train,y_train)
    
    return xgb_reg

@XGBoostRegression.route("/train_xgboost_regressor", methods = ["GET","POST"])
def train_xgboost_regressor():
    global xgboost_regressor
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    
    xgboost_regressor=xgboost_regression(X_train,y_train)
    return render_template("models/Boosting/Regressors/XgboostRegressor.html",
                           training=X_train.shape, target=X_test.shape,train_status="Model is trained Successfully",
                           columns=training,model="xgboost_reg")
    
@XGBoostRegression.route("/test_xgboost_regressor", methods = ["GET","POST"])
def test_xgboost_regressor():
    
    from ModelsPy.Accuracies import check_r2_score
    
    score=check_r2_score(y_test,xgboost_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@XGBoostRegression.route("/predict_xgboost_reg", methods = ["GET","POST"])
def predict_xgboost_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = xgboost_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "XGBoost Regression", prediction=score[0])
