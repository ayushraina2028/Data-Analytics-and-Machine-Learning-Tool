from flask import Blueprint, render_template, request, jsonify
SupportVectorClassification = Blueprint('SupportVectorClassification', __name__, template_folder='templates', static_folder='static')

#Support Vector Classification
def support_vector_classification(X_train,y_train,random_state,max_iter,kernel,parameter,gamma):
    
    from sklearn.svm import SVC
    svc = SVC(kernel=kernel, C=parameter, gamma=gamma, random_state=random_state, max_iter=max_iter)
    svc.fit(X_train,y_train)
    
    return svc

@SupportVectorClassification.route("/train_support_vector_classifier", methods = ["GET","POST"])
def train_support_vector_classifier():
    global support_vector_classifier
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
   
    random_state = request.form.get("random_state")
    max_iter = request.form.get("max_iter")
    kernel = request.form.get("kernel")
    parameter = request.form.get("parameter")
    gamma = request.form.get("gamma")
    
    if not random_state:
        random_state=None
    else:
        random_state = int(random_state)
        
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
    
    support_vector_classifier = support_vector_classification(X_train,y_train,random_state=random_state, max_iter=max_iter, kernel=kernel, parameter=parameter, gamma=gamma)
    return render_template("models/SupportVectorMachines/SupportVectorClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="svc")

@SupportVectorClassification.route("/test_support_vector_classifier", methods = ["GET","POST"])
def test_support_vector_classifier():
    
    from ModelsPy.Accuracies import check_accuracy
    
    score=check_accuracy(y_test,support_vector_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@SupportVectorClassification.route("/predict_svc", methods = ["GET","POST"])
def predict_svc():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = support_vector_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Support Vector Classifier", prediction=score[0])

