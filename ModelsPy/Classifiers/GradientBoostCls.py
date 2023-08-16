from flask import Blueprint, render_template, request, jsonify
GradientBoostClassification = Blueprint('GradientBoostClassification', __name__, template_folder='templates', static_folder='static')

# Gradient Boost Classification
def gradientboost_classification(X_train,y_train,n_estimators,learning_rate,max_depth,criterion):
    
    from sklearn.ensemble import GradientBoostingClassifier
    gradient_boost = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,criterion=criterion)
    gradient_boost.fit(X_train,y_train)
    
    return gradient_boost

@GradientBoostClassification.route("/train_gradientboost_classifier", methods = ["GET","POST"])
def train_gradientboost_classifier():
    global gradientboost_classifier
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    
    n_estimators = request.form.get("n_estimators")
    learning_rate = request.form.get("learning_rate")
    max_depth = request.form.get("max_depth")
    criterion = request.form.get("criterion")
    
    
    if not n_estimators:
        n_estimators=100
    else:
        n_estimators = int(n_estimators)
        
    if not learning_rate:
        learning_rate=0.1
    else:
        learning_rate = float(learning_rate)
        
    if not max_depth:
        max_depth=3
    else:
        max_depth = int(max_depth)
        
    if not criterion:
        criterion="friedman_mse"
    
    
    gradientboost_classifier=gradientboost_classification(X_train,y_train,n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, criterion=criterion)
    return render_template("models/Boosting/Classifiers/GradientBoostClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="gradient_boosting_cls")

@GradientBoostClassification.route("/test_gradientboost_classifier", methods = ["GET","POST"])
def test_gradientboost_classifier():
    
    from ModelsPy.Accuracies import check_accuracy
    
    score=check_accuracy(y_test,gradientboost_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@GradientBoostClassification.route("/predict_gradient_boosting_cls", methods = ["GET","POST"])
def predict_gradient_boosting_cls():
    
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = gradientboost_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Gradient Boosting Classifier", prediction=score[0])
