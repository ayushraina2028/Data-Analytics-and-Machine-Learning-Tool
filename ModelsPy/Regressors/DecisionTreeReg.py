from flask import Blueprint, render_template, request, jsonify
DecisionTreeRegression = Blueprint('DecisionTreeRegression', __name__, template_folder='templates', static_folder='static')


# Decision Tree Regression
def decision_tree_regression(X_train,y_train):
    
    from sklearn.tree import DecisionTreeRegressor
    tree=DecisionTreeRegressor(random_state=42)
    tree.fit(X_train,y_train)
    
    return tree

# Extra Tree Regression
def extra_tree_regression(X_train,y_train):
    
    from sklearn.tree import ExtraTreeRegressor
    trees=ExtraTreeRegressor(random_state=42)
    trees.fit(X_train,y_train)
    
    return trees


@DecisionTreeRegression.route("/train_decision_tree_reg", methods = ["GET","POST"])
def train_decision_tree_reg():
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training

    global decision_tree_regressor
    tree=request.form.get("tree")
    
    if tree == "ExtraTreeRegressor":
        decision_tree_regressor=extra_tree_regression(X_train,y_train)
        return render_template("models/DecisionTree/DecisionTreeRegressor.html",
                            target=target, trains=training,train_status="ExtraTree is trained Successfully")    
    else:
        decision_tree_regressor=decision_tree_regression(X_train,y_train)
        return render_template("models/DecisionTree/DecisionTreeRegressor.html",
                            target=target, trains=training,train_status="Model is trained Successfully",
                            columns=training,model="decision_tree_reg")
        
@DecisionTreeRegression.route("/test_decision_tree_reg", methods = ["GET","POST"])
def test_decision_tree_reg():
    
    from ModelsPy.Accuracies import check_r2_score
    
    score=check_r2_score(y_test,decision_tree_regressor.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@DecisionTreeRegression.route("/predict_decision_tree_reg", methods = ["GET","POST"])
def predict_decision_tree_reg():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = decision_tree_regressor.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Decision Tree Regression", prediction=score[0])
