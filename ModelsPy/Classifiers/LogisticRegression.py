from flask import Blueprint, render_template, request, jsonify
LogisticRegression = Blueprint('LogisticRegression', __name__, template_folder='templates', static_folder='static')

# Logistic Regression
def logistic_regression(X_train,y_train,types, random_state, max_iter, multiclass, bias, solver):
    
    if bias == "None":
        bias = None
    
    if types=="Binary":
        from sklearn.linear_model import LogisticRegression
        log_reg=LogisticRegression(random_state=random_state, max_iter=max_iter, penalty=bias, solver=solver)
        log_reg.fit(X_train,y_train)
        return log_reg
   
    if types=="MultiClass":
        from sklearn.linear_model import LogisticRegression
        log_reg=LogisticRegression(random_state=random_state, max_iter=max_iter, multi_class=multiclass, penalty=bias, solver=solver)
        log_reg.fit(X_train,y_train)
        return log_reg
    
@LogisticRegression.route("/train_logistic_regression_classifier", methods = ["GET","POST"])
def train_logistic_regression_classifier():
    global logistic_regression_classifier
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    
    classify = request.form.get("logistic")
    random_state = request.form.get("random_state")
    max_iter = request.form.get("max_iter")
    multiclass = request.form.get("multiclass")
    bias = request.form.get("bias")
    solver = request.form.get("solver")
    
    
    if not random_state:
        random_state=None
    else:
        random_state = int(random_state)
        
        
    if not max_iter:
        max_iter=100
    else:
        max_iter = int(max_iter)

    if not multiclass:
        multiclass="auto"
    if not bias:
        bias = "l2"
    if not solver:
        solver="lbfgs"
        
    
    
    logistic_regression_classifier=logistic_regression(X_train,y_train,types=classify, random_state=random_state, max_iter=max_iter, multiclass=multiclass, bias=bias, solver=solver)
    return render_template("models/LogisticalRegression/Logistic.html",
                            target=target, trains=training,train_status=f"{classify} Logistic Model is trained Successfully",
                            columns=training,model="logistic_cls")
    

@LogisticRegression.route("/test_logistical_regression_classifier", methods = ["GET","POST"])
def test_logistical_regression_classifier():
    
    from ModelsPy.Accuracies import check_accuracy
    
    score=check_accuracy(y_test,logistic_regression_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})


# Prediction classification all models
@LogisticRegression.route("/predict_logistic_cls", methods = ["GET","POST"])
def predict_logistic_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = logistic_regression_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Logistic Regression", prediction=score[0])
