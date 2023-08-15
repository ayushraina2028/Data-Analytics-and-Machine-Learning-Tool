from flask import Blueprint, render_template, request, redirect, url_for
modelPage = Blueprint('modelPage', __name__, template_folder='templates', static_folder='static')

@modelPage.route("/start_machine", methods = ["GET","POST"])
def start_machine():
    global df
    from StartersPy.Encoding import df
    
    global X_train,X_test,y_train,y_test,training,target
    
    test=request.form.get("test_size")
    problem=request.form.get("problem")
    
    target = request.form.getlist('columns')
    target = [i.replace(","," ") for i in target]
    target=target[0]
    
    training = request.form.getlist('columns1')
    training = [i.replace(","," ") for i in training]
    
    # Separating Independent and Dependent Features
    X = df[training]
    y=df[target]
    
    
    #splitting
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=float(test),random_state=42)
    
    
    if problem=="Regression":
        return render_template("ML/regression.html", test_size=test,
                               training=X_train.shape, testing=X_test.shape)
    else:
        return render_template("ML/classification.html", test_size=test,
                               training=X_train.shape, testing=X_test.shape)
    
@modelPage.route("/train_cls_models", methods = ["GET","POST"])
def train_cls_models():
    global classification_models
    classification_models=request.form.getlist("classification_models")
    
    if len(classification_models) > 1:
        return render_template("ML/classification2.html",training=X_train.shape, testing=X_test.shape)
    
    for i in classification_models:
        
        if i == "decision_tree_cls":
            return render_template("models/DecisionTree/DecisionTreeClassifier.html",
                                   target=target, trains=training)
        if i== "logistic":
            return render_template("models/LogisticalRegression/Logistic.html",
                                      target=target, trains=training)
        if i== "naive_bayes":
            return render_template("models/NaiveBayes/NaiveBayes.html",
                                      target=target, trains=training)
        if i == "svc":
            return render_template("models/SupportVectorMachines/SupportVectorClassifier.html",
                                      target=target, trains=training)
        if i == "random_forest_cls":
            return render_template("models/RandomForest/RandomForestClassifier.html",
                                      target=target, trains=training)
        if i == "adaboost":
            return render_template("models/Boosting/Classifiers/AdaboostClassifier.html",
                                      target=target, trains=training)
        if i == "gradientboost":
            return render_template("models/Boosting/Classifiers/GradientBoostClassifier.html",
                                      target=target, trains=training)
        if i == "knn_cls":
            return render_template("models/KNearestNeighbours/KNNClassifier.html",
                                      target=target, trains=training)
@modelPage.route("/train_reg_models", methods = ["GET","POST"])
def train_reg_models():
    
    
    global regression_models
    regression_models=request.form.getlist("regression_models")
    
    if len(regression_models) > 1:
        return render_template("ML/regression2.html",training=X_train.shape, testing=X_test.shape)
    
    for i in regression_models:
        
        if i == "linear_reg":
            return render_template("models/LinearRegression/LinearRegression.html",
                                   target=target, trains=training)
        if i == "decision_tree_reg":
            return render_template("models/DecisionTree/DecisionTreeRegressor.html",
                                   target=target, trains=training)
        if i == "svr":
            return render_template("models/SupportVectorMachines/SupportVectorRegressor.html",
                                   target=target, trains=training)
        if i == "random_forest_reg":
            return render_template("models/RandomForest/RandomForestRegressor.html",
                                   target=target, trains=training)
        if i == "adaboost_reg":
            return render_template("models/Boosting/Regressors/AdaboostRegressor.html",
                                   target=target, trains=training)
        if i == "gradientboost_reg":
            return render_template("models/Boosting/Regressors/GradientBoostRegressor.html",
                                   target=target, trains=training)
        if i == "xgboost_reg":
            return render_template("models/Boosting/Regressors/XgboostRegressor.html",
                                   target=target, trains=training)
            
        if i == "knn_reg":
            return render_template("models/KNearestNeighbours/KNNRegressor.html",
                                   target=target, trains=training)
        
@modelPage.route("/test_cls_models", methods = ["GET","POST"])
def test_cls_models():
    
    for i in classification_models:
        
        if i == "logistic":
            
            if len(y_train.unique()) > 2:
                log_cls = LogisticRegression(multi_class="ovr")
                log_cls.fit(X_train,y_train)
                accuracy_logistic = check_accuracy(y_test,log_cls.predict(X_test))
                accuracy_logistic=accuracy_logistic*100
            else:
                log_cls = LogisticRegression()
                log_cls.fit(X_train,y_train)
                accuracy_logistic = check_accuracy(y_test,log_cls.predict(X_test))
                accuracy_logistic=accuracy_logistic*100
            
        elif i == "decision_tree_cls":
            dt_cls = DecisionTreeClassifier()
            dt_cls.fit(X_train,y_train)
            accuracy_decision_tree_cls=check_accuracy(y_test,dt_cls.predict(X_test))
            accuracy_decision_tree_cls=accuracy_decision_tree_cls*100
            
        elif i == "naive_bayes":
            
            if len(y_train.unique()) > 2:
                nb_cls = MultinomialNB()
                nb_cls.fit(X_train,y_train)
                accuracy_naive_bayes=check_accuracy(y_test,nb_cls.predict(X_test))
                accuracy_naive_bayes=accuracy_naive_bayes*100
            else:
                nb_cls = GaussianNB()
                nb_cls.fit(X_train,y_train)
                accuracy_naive_bayes=check_accuracy(y_test,nb_cls.predict(X_test))
                accuracy_naive_bayes=accuracy_naive_bayes*100
        
        elif i == "svc":
            svc_cls = SVC()
            svc_cls.fit(X_train,y_train)
            accuracy_svc=check_accuracy(y_test,svc_cls.predict(X_test))
            accuracy_svc=accuracy_svc*100
        
        elif i == "random_forest_cls":
            rf_cls = RandomForestClassifier()
            rf_cls.fit(X_train,y_train)
            accuracy_random_forest_cls=check_accuracy(y_test,rf_cls.predict(X_test))
            accuracy_random_forest_cls=accuracy_random_forest_cls*100
            
        elif i == "adaboost":
            adaboost_cls = AdaBoostClassifier()
            adaboost_cls.fit(X_train,y_train)
            accuracy_adaboost=check_accuracy(y_test,adaboost_cls.predict(X_test))
            accuracy_adaboost=accuracy_adaboost*100
            
        elif i == "gradientboost":
            gradientboost_cls = GradientBoostingClassifier()
            gradientboost_cls.fit(X_train,y_train)
            accuracy_gradientboost=check_accuracy(y_test,gradientboost_cls.predict(X_test))
            accuracy_gradientboost=accuracy_gradientboost*100
            
        elif i == "knn_cls":
            knn_cls = KNeighborsClassifier()
            knn_cls.fit(X_train,y_train)
            accuracy_knn_cls=check_accuracy(y_test,knn_cls.predict(X_test))
            accuracy_knn_cls=accuracy_knn_cls*100
            
    return render_template("ML/classification2.html",training=X_train.shape, testing=X_test.shape,
                               accuracy_logistic=accuracy_logistic,
                               accuracy_decision_tree_cls=accuracy_decision_tree_cls,
                               accuracy_naive_bayes=accuracy_naive_bayes,
                               accuracy_svc=accuracy_svc,
                               accuracy_random_forest_cls=accuracy_random_forest_cls,
                               accuracy_adaboost=accuracy_adaboost,
                               accuracy_gradientboost=accuracy_gradientboost,
                               accuracy_knn_cls=accuracy_knn_cls)

@modelPage.route("/test_reg_models", methods = ["GET","POST"])
def test_reg_models():
    
    for i in regression_models:
        
        if i == "linear_reg":
            lin_reg = LinearRegression()
            lin_reg.fit(X_train,y_train)
            accuracy_linear_reg = check_r2_score(y_test,lin_reg.predict(X_test))
            accuracy_linear_reg = accuracy_linear_reg*100
            
        elif i == "decision_tree_reg":
            dt_reg = DecisionTreeRegressor()
            dt_reg.fit(X_train,y_train)
            accuracy_decision_tree_reg=check_r2_score(y_test,dt_reg.predict(X_test))
            accuracy_decision_tree_reg=accuracy_decision_tree_reg*100
            
        elif i == "svr":
            svr = SVR()
            svr.fit(X_train,y_train)
            accuracy_svr=check_r2_score(y_test,svr.predict(X_test))
            accuracy_svr=accuracy_svr*100
            
        elif i == "random_forest_reg":
            rf_reg = RandomForestRegressor()
            rf_reg.fit(X_train,y_train)
            accuracy_random_forest_reg=check_r2_score(y_test,rf_reg.predict(X_test))
            accuracy_random_forest_reg=accuracy_random_forest_reg*100
            
        elif i == "adaboost_reg":
            ada_reg = AdaBoostRegressor()
            ada_reg.fit(X_train,y_train)
            accuracy_adaboost_reg=check_r2_score(y_test,ada_reg.predict(X_test))
            accuracy_adaboost_reg=accuracy_adaboost_reg*100
            
        elif i == "gradientboost_reg":
            gb_reg = GradientBoostingRegressor()
            gb_reg.fit(X_train,y_train)
            accuracy_gradient_boost_reg=check_r2_score(y_test,gb_reg.predict(X_test))
            accuracy_gradient_boost_reg=accuracy_gradient_boost_reg*100
            
        elif i == "xgboost_reg":
            xgb_reg = xbs.XGBRegressor()
            xgb_reg.fit(X_train,y_train)
            accuracy_xgboost_reg=check_r2_score(y_test,xgb_reg.predict(X_test))
            accuracy_xgboost_reg=accuracy_xgboost_reg*100
            
        elif i == "knn_reg":
            knn_reg = KNeighborsRegressor()
            knn_reg.fit(X_train,y_train)
            accuracy_knn_reg=check_r2_score(y_test,knn_reg.predict(X_test))
            accuracy_knn_reg=accuracy_knn_reg*100
    
    return render_template("ML/regression2.html",training=X_train.shape, testing=X_test.shape,
                           accuracy_linear_reg=accuracy_linear_reg,
                           accuracy_decision_tree_reg=accuracy_decision_tree_reg,
                           accuracy_svr=accuracy_svr,
                           accuracy_random_forest_reg=accuracy_random_forest_reg,
                           accuracy_adaboost_reg=accuracy_adaboost_reg,
                           accuracy_gradient_boost_reg=accuracy_gradient_boost_reg,
                           accuracy_xgboost_reg=accuracy_xgboost_reg,
                           accuracy_knn_reg=accuracy_knn_reg)
            
