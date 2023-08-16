from flask import Blueprint, render_template, request, jsonify
SupportVectorRegression = Blueprint('SupportVectorRegression', __name__, template_folder='templates', static_folder='static')

#Support Vector Regression
def support_vector_regression(X_train,y_train,epsilon,max_iter,kernel,parameter,gamma):
    
    from sklearn.svm import SVR
    svr = SVR(kernel=kernel,epsilon=epsilon, C=parameter, gamma=gamma, max_iter=max_iter)
    svr.fit(X_train,y_train)
    
    return svr

@SupportVectorRegression.route("/train_support_vector_regressor", methods = ["GET","POST"])
def train_support_vector_regressor():
    global support_vector_regressor
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
 
   
    epsilon = request.form.get("epsilon")
    max_iter = request.form.get("max_iter")
    kernel = request.form.get("kernel")
    parameter = request.form.get("parameter")
    gamma = request.form.get("gamma")
    
    if not epsilon:
        epsilon=0.1
    else:
        epsilon = float(epsilon)
        
    if not max_iter:
        max_iter=-1
    else:
        max_iter = int(max_iter)

    if not kernel:
        kernel = "rbf"
        
    if not parameter:
        parameter = 1.0
    else:
        parameter = float(parameter)
        
    if not gamma:
        gamma = "scale"
    elif gamma == "auto":
        gamma = "auto"
    else:
        gamma = float(gamma)
    
    support_vector_regressor = support_vector_regression(X_train,y_train,epsilon=epsilon, max_iter=max_iter, kernel=kernel, parameter=parameter, gamma=gamma)
    return render_template("models/SupportVectorMachines/SupportVectorRegressor.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="support_vector_reg")

@SupportVectorRegression.route("/test_support_vector_regressor", methods = ["GET","POST"])
def test_support_vector_regressor():
    
    from ModelsPy.Accuracies import check_r2_score
    
    score=check_r2_score(y_test,support_vector_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@SupportVectorRegression.route("/predict_support_vector_reg", methods = ["GET","POST"])
def predict_svr():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = support_vector_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Support Vector Regression", prediction=score[0])
