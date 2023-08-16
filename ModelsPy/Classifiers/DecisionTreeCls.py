from flask import Blueprint, render_template, request, jsonify
DecisionTreeClassification = Blueprint('DecisionTreeClassification', __name__, template_folder='templates', static_folder='static')

# Decision Tree Classification
def decision_tree_classification(X_train,y_train):
    
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_train,y_train)
    
    return tree

# Extra Tree Classification
def extra_tree_classification(X_train,y_train):
    
    from sklearn.tree import ExtraTreeClassifier
    trees=ExtraTreeClassifier(random_state=42)
    trees.fit(X_train,y_train)
    
    return trees
    


@DecisionTreeClassification.route("/train_decision_tree_classifier", methods = ["GET","POST"])
def train_decision_tree_classifier():
    global decision_tree_classifier
    tree = request.form.get("tree")
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    
    if tree == "ExtraTreeClassifier":
        decision_tree_classifier=extra_tree_classification(X_train,y_train)
        return render_template("models/DecisionTree/DecisionTreeClassifier.html",
                                target=target, trains=training,train_status="Extra Tree Model is trained Successfully",
                                columns=training,model="decision_tree_cls")
    else:
        decision_tree_classifier=decision_tree_classification(X_train,y_train)
        return render_template("models/DecisionTree/DecisionTreeClassifier.html",
                                target=target, trains=training,train_status="Model is trained Successfully",
                                columns=training,model="decision_tree_cls")

@DecisionTreeClassification.route("/test_decision_tree_classifier", methods = ["GET","POST"])
def test_decision_tree_classifier():
    
    from ModelsPy.Accuracies import check_accuracy
    
    score=check_accuracy(y_test,decision_tree_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@DecisionTreeClassification.route("/predict_decision_tree_cls", methods = ["GET","POST"])
def predict_decision_tree_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = decision_tree_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Decision Tree Classifier", prediction=score[0])
