from flask import Blueprint, render_template, request, jsonify
LinearRegression = Blueprint('LinearRegression', __name__, template_folder='templates', static_folder='static')
import matplotlib.pyplot as plt

# Linear Regression
def linear_regression(X_train,y_train):
    
    from sklearn.linear_model import LinearRegression
    linear_regressor=LinearRegression()
    linear_regressor.fit(X_train,y_train)
    
    return linear_regressor

# Ridge Regression (L2 Regularization)
def ridge_regression(X_train,y_train):
    
    from sklearn.linear_model import Ridge
    ridge_regressor=Ridge()
    ridge_regressor.fit(X_train,y_train)
    
    return ridge_regressor

# Lasso Regression (L1 Regularization)
def lasso_regression(X_train,y_train):
    
    from sklearn.linear_model import Lasso
    lasso_regressor=Lasso()
    lasso_regressor.fit(X_train,y_train)
    
    return lasso_regressor


# Elastic NET  Regression (L1 + L2 Regularization)
def elastic_net_regression(X_train,y_train):
    
    from sklearn.linear_model import ElasticNet
    elastic_net_regressor=ElasticNet()
    elastic_net_regressor.fit(X_train,y_train)
    
    return elastic_net_regressor


@LinearRegression.route("/train_linear_reg", methods = ["GET","POST"])
def train_linear_reg():
    global linear_regressor
    bias=request.form.get("bias")
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    # is_scale=request.form.get("scaler")
    # if is_scale=="yes":
    #     X_train,X_test=scale_down(X_train,X_test)
    # Above piece of Code is not Working and i do not know why
    
    if bias == "L1 Regularization":
        linear_regressor=lasso_regression(X_train,y_train)
        return render_template("models/LinearRegression/LinearRegression.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           message="Click Here",columns = training,model = "linear_reg")
        
    elif bias == "L2 Regularization":
        linear_regressor=ridge_regression(X_train,y_train)
        return render_template("models/LinearRegression/LinearRegression.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           message="Click Here",columns = training,model = "linear_reg")
    elif bias == "Both":
        linear_regressor=ridge_regression(X_train,y_train)
        return render_template("models/LinearRegression/LinearRegression.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           message="Click Here",columns = training,model = "linear_reg")
    else:
        linear_regressor=linear_regression(X_train,y_train)
        return render_template("models/LinearRegression/LinearRegression.html",
                            target=target, trains=training,train_status="Model is trained Successfully",
                            columns = training,model = "linear_reg")

@LinearRegression.route("/test_linear_reg", methods = ["GET","POST"])
def test_linear_reg():
    
    from ModelsPy.Accuracies import check_r2_score
    
    score=check_r2_score(y_test,linear_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})
                   
@LinearRegression.route("/visualize_linear_reg", methods = ["GET","POST"])
def visualize_linear_reg():
    plt.clf()
    plt.figure(figsize=(15,15))
    plt.scatter(X_train,y_train,color="red",s=2)
    plt.plot(X_train,linear_regressor.predict(X_train),color="blue")
    plt.title("Linear Regression")
    plt.xlabel("Independent Variable")
    plt.ylabel("Dependent Variable")
    plt.savefig("static/images/models/LinearRegression/linear_reg.png", bbox_inches = 'tight')
    
    return render_template("models/LinearRegression/LinearRegression2.html",
                           graph1="static/images/models/LinearRegression/linear_reg.png")

# Predictions Regressions
@LinearRegression.route("/predict_linear_reg", methods = ["GET","POST"])
def predict_linear_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = linear_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Linear Regression", prediction=score[0])    
