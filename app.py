# Required Dependencies
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import io
import warnings
warnings.filterwarnings("ignore")

# Importing Blue Print
from StartersPy.starter import starter
from StartersPy.Phase1  import phase1
from StartersPy.missvalue import missvalue
from StartersPy.Encoding import encoder
from StartersPy.TrainTestSplit import trainTest
from ModelsPy.modelPage import modelPage
from ModelsPy.Regressors.LinearRegression import LinearRegression
from ModelsPy.Regressors.DecisionTreeReg import DecisionTreeRegression
from ModelsPy.Regressors.SupportVectorReg import SupportVectorRegression
from ModelsPy.Regressors.RandomForest import RandomForestRegression
from ModelsPy.Regressors.AdaboostReg import AdaboostRegression
from ModelsPy.Regressors.GradientBoostReg import GradientBoostRegression
from ModelsPy.Regressors.XGBoostReg import XGBoostRegression
from ModelsPy.Regressors.KNNReg import KNNRegression

from ModelsPy.Classifiers.LogisticRegression import LogisticRegression



# Registering Blue Prints
app = Flask(__name__)
app.register_blueprint(starter)
app.register_blueprint(phase1)
app.register_blueprint(missvalue)
app.register_blueprint(encoder)
app.register_blueprint(trainTest)
app.register_blueprint(modelPage)
app.register_blueprint(LinearRegression)
app.register_blueprint(LogisticRegression)
app.register_blueprint(DecisionTreeRegression)
app.register_blueprint(SupportVectorRegression)
app.register_blueprint(RandomForestRegression)
app.register_blueprint(AdaboostRegression)
app.register_blueprint(GradientBoostRegression)
app.register_blueprint(XGBoostRegression)
app.register_blueprint(KNNRegression)



matplotlib.use('Agg')
plt=matplotlib.pyplot



# To scale down training data
def scale_down(X_train, X_test):
    
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    return X_train, X_test

# To check accuracy of the regression models
def check_r2_score(y_test, y_pred):
    
    from sklearn.metrics import r2_score
    score = r2_score(y_test, y_pred)
    
    return score

# To check accuracy of classification models
def check_accuracy(y_test, y_pred):
    
    from sklearn.metrics import accuracy_score
    score = accuracy_score(y_test, y_pred)
    
    return score


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
    

    
#Naive Bias Classifier
def naive_bayes_classifier(X_train,y_train,types):
        
        if types=="Gaussian":
            from sklearn.naive_bayes import GaussianNB
            naive=GaussianNB()
            naive.fit(X_train,y_train)
            return naive
        
        if types=="Multinomial":
            from sklearn.naive_bayes import MultinomialNB
            naive=MultinomialNB()
            naive.fit(X_train,y_train)
            return naive
        
        if types=="Bernoulli":
            from sklearn.naive_bayes import BernoulliNB
            naive=BernoulliNB()
            naive.fit(X_train,y_train)
            return naive
        
        if types=="Complement":
            from sklearn.naive_bayes import ComplementNB
            naive=ComplementNB()
            naive.fit(X_train,y_train)
            return naive
        
        if types=="Categorical":
            from sklearn.naive_bayes import CategoricalNB
            naive=CategoricalNB()
            naive.fit(X_train,y_train)
            return naive
        
#Support Vector Classification
def support_vector_classification(X_train,y_train,random_state,max_iter,kernel,parameter,gamma):
    
    from sklearn.svm import SVC
    svc = SVC(kernel=kernel, C=parameter, gamma=gamma, random_state=random_state, max_iter=max_iter)
    svc.fit(X_train,y_train)
    
    return svc



# Random Forest Classification
def random_forest_classification(X_train,y_train,n_estimators,max_depth,max_features,criterion,bootstrap,oob_score):
    
    if max_depth == "None":
        max_depth = None 
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score)
    forest.fit(X_train,y_train)
    
    return forest


# Adaboost Classification
def adaboost_classification(X_train,y_train,n_estimators,learning_rate,algorithm):
    
    from sklearn.ensemble import AdaBoostClassifier
    adaboost = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)
    adaboost.fit(X_train,y_train)
    
    return adaboost
    
# Gradient Boost Classification
def gradientboost_classification(X_train,y_train,n_estimators,learning_rate,max_depth,criterion):
    
    from sklearn.ensemble import GradientBoostingClassifier
    gradient_boost = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,criterion=criterion)
    gradient_boost.fit(X_train,y_train)
    
    return gradient_boost



# Gradient Boost Regressor
def gradient_boost_regression(X_train,y_train,n_estimators, learning_rate, loss, criterion, max_depth, max_features):
    
    from sklearn.ensemble import GradientBoostingRegressor
    gradient_boost = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, criterion=criterion, max_depth=max_depth, max_features=max_features)
    gradient_boost.fit(X_train,y_train)
    
    return gradient_boost



# KNN Classifier
def knn_classification(X_train,y_train,n_neighbors,weights,algorithm,leaf_size,p):
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
    knn.fit(X_train,y_train)
    
    return knn


            

@app.route("/download")
def download():
    
    global send,df
    from StartersPy.Encoding import df
    
    csv_file=io.StringIO()
    df.to_csv(csv_file,index=False)
    
    return send_file(
        io.BytesIO(csv_file.getvalue().encode()),
        as_attachment=True,
        download_name="Dataset.csv",
        mimetype='text/csv'
    )

# @app.route("/phase4")
# def phase4():
#     return render_template("EDA.html")


# @app.route("/phase5")
# def phase5():
    
#     return render_template("ML_intro.html")
    

    










  


@app.route("/train_naive_bayes_classifier", methods = ["GET","POST"])
def train_native_bayes_classifier():
    global native_bayes_classifier
    classify = request.form.get("algos")
    native_bayes_classifier=naive_bayes_classifier(X_train,y_train,types=classify)
    return render_template("models/NaiveBayes/NaiveBayes.html",
                            target=target, trains=training,train_status=f"{classify} Naive Bayes Model is trained Successfully",
                            columns=training,model="naive_bayes")    

@app.route("/test_naive_bayes_classifier", methods = ["GET","POST"])
def test_native_bayes_classifier():
        
        score=check_accuracy(y_test,native_bayes_classifier.predict(X_test))
        score=score*100
        return jsonify({"score":score})

@app.route("/train_decision_tree_classifier", methods = ["GET","POST"])
def train_decision_tree_classifier():
    global decision_tree_classifier
    tree = request.form.get("tree")
    
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

@app.route("/test_decision_tree_classifier", methods = ["GET","POST"])
def test_decision_tree_classifier():
    
    score=check_accuracy(y_test,decision_tree_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_support_vector_classifier", methods = ["GET","POST"])
def train_support_vector_classifier():
    global support_vector_classifier
    
   
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

@app.route("/test_support_vector_classifier", methods = ["GET","POST"])
def test_support_vector_classifier():
    
    score=check_accuracy(y_test,support_vector_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_random_forest_classifier", methods = ["GET","POST"])
def train_random_forest_classifier():
    global random_forest_classifier
    
    n_estimators = request.form.get("n_estimators")
    max_depth = request.form.get("max_depth")
    max_features = request.form.get("max_features")
    criterion = request.form.get("criterion")
    bootstrap = request.form.get("bootstrap")
    oob_score = request.form.get("oob")
    
    
    if not n_estimators:
        n_estimators=100
    else:
        n_estimators = int(n_estimators)
        
    if not max_depth:
        max_depth=None
    else: 
        max_depth = int(max_depth)
        
    if not max_features:
        max_features="sqrt"
    elif max_features == "log2":
        max_features = "log2"
    elif max_features == "None":
        max_features = None
    else:
        max_features = float(max_features)
        
    if not criterion:
        criterion="gini"
    
    if not bootstrap:
        bootstrap=True
    else:
        bootstrap=False
        
    if not oob_score:
        oob_score=False
    else:
        oob_score=True
        
    random_forest_classifier=random_forest_classification(X_train,y_train,n_estimators=n_estimators, max_depth=max_depth,max_features=max_features, criterion=criterion, bootstrap=bootstrap, oob_score=oob_score)
    return render_template("models/RandomForest/RandomForestClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="random_forest_cls")

@app.route("/test_random_forest_classifier", methods = ["GET","POST"])
def test_random_forest_classifier():
    
    score=check_accuracy(y_test,random_forest_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_adaboost_classifier", methods = ["GET","POST"])
def train_adaboost_classifier():
    global adaboost_classifier
    
    n_estimators = request.form.get("n_estimators")
    learning_rate = request.form.get("learning_rate")
    algorithm = request.form.get("algorithm")
    
    if not n_estimators:
        n_estimators=50
    else:
        n_estimators = int(n_estimators)
        
    if not learning_rate:
        learning_rate=1.0
    else:
        learning_rate = float(learning_rate)
        
    if not algorithm:
        algorithm="SAMME.R"
    
    adaboost_classifier=adaboost_classification(X_train,y_train,n_estimators=n_estimators, learning_rate=learning_rate, algorithm=algorithm)
    return render_template("models/Boosting/Classifiers/AdaboostClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="adaboost_cls")

@app.route("/test_adaboost_classifier", methods = ["GET","POST"])
def test_adaboost_classifier():
    
    score=check_accuracy(y_test,adaboost_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_gradientboost_classifier", methods = ["GET","POST"])
def train_gradientboost_classifier():
    global gradientboost_classifier
    
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

@app.route("/test_gradientboost_classifier", methods = ["GET","POST"])
def test_gradientboost_classifier():
    
    score=check_accuracy(y_test,gradientboost_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})

@app.route("/train_knn_classifier", methods = ["GET","POST"])
def train_knn_classifier():
    global knn_classifier
    
    n_neighbors = request.form.get("n_neighbours")
    weights = request.form.get("weights")
    algorithm = request.form.get("algorithm")
    leaf_size = request.form.get("leaf_size")
    p = request.form.get("p")
    
    if not n_neighbors:
        n_neighbors=5
    else:
        n_neighbors = int(n_neighbors)
        
    if not weights:
        weights="uniform"
    
        
    if not algorithm:
        algorithm="auto"
        
    if not leaf_size:
        leaf_size=30
    else:
        leaf_size = int(leaf_size)
        
    if not p:
        p=2
    else:
        p = int(p)
        
    knn_classifier=knn_classification(X_train,y_train,n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p)
    return render_template("models/KNearestNeighbours/KNNClassifier.html",
                           target=target, trains=training,train_status="Model is trained Successfully",
                           columns=training,model="knn_cls")

@app.route("/test_knn_classifier", methods = ["GET","POST"])
def test_knn_classifier():
    
    score=check_accuracy(y_test,knn_classifier.predict(X_test))
    score=score*100
    return jsonify({"score":score})
    

@app.route("/graph", methods = ["GET","POST"])
def grapher():

    global send,df
    from StartersPy.Encoding import df
    
    columns = df.columns.to_list()
    return render_template("/Graph/main.html",columns=columns)

@app.route("/plot_graph", methods = ["GET","POST"])
def show_graph():
    
    feature11 = request.form.get("feature11")
    feature12 = request.form.get("feature12")
    feature21 = request.form.get("feature21")
    feature22 = request.form.get("feature22")
    
    if feature11:
        feature11 = feature11.replace(","," ")
    
    if feature12:
        feature12 = feature12.replace(","," ")
        
    if feature21:
        feature21 = feature21.replace(","," ")
        
    if feature22:
        feature22 = feature22.replace(","," ")
        

    plot1 = request.form.get("columns1")
    plot2 = request.form.get("columns2")
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))
    
    if plot1 == "scatter plot":
        axes[0].scatter(df[feature11],df[feature12])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
        
    elif plot1 == "line plot":
        axes[0].plot(df[feature11],df[feature12])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
        
    elif plot1 == "bar plot":
        axes[0].bar(df[feature11],df[feature12])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
        
    elif plot1 == "box plot":
        axes[0].boxplot(df[feature11])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
    
    elif plot1 == "violin plot":
        axes[0].violinplot(df[feature11])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)   
        
    elif plot1 == "heatmap":
        df1 = df[[feature11,feature12]]
        sns.heatmap(df1.corr(),annot=True, ax = axes[0])
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
    
    elif plot1 == "hexbin":
        left_hex = axes[0].hexbin(df[feature11],df[feature12],gridsize=20, cmap = "Blues")
        axes[0].set_title(f"Hexbin plot of {feature11} and {feature12}")
        plt.colorbar(left_hex, ax=axes[0], label = "Density")
        axes[0].set_xlabel(feature11)
        axes[0].set_ylabel(feature12)
        
    
        
        
    # Same for plot 2
    if plot2 == "scatter plot":
        axes[1].scatter(df[feature21],df[feature22])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        
    elif plot2 == "line plot":
        axes[1].plot(df[feature21],df[feature22])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        
    elif plot2 == "bar plot":
        axes[1].bar(df[feature21],df[feature22])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        
    elif plot2 == "box plot":
        axes[1].boxplot(df[feature21])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
    
    elif plot2 == "violin plot":
        axes[1].violinplot(df[feature21])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        
    elif plot2 == "heatmap":
        df2 = df[[feature21,feature22]]
        sns.heatmap(df2.corr(),annot=True, ax = axes[1])
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        
    elif plot2 == "hexbin":
        right_hex = axes[1].hexbin(df[feature21],df[feature22],gridsize=20, cmap = "Reds")
        axes[1].set_title(f"Hexbin plot of {feature21} and {feature22}")
        axes[1].set_xlabel(feature21)
        axes[1].set_ylabel(feature22)
        plt.colorbar(right_hex, ax=axes[1], label = "Density")
        
        
    
    
    plt.savefig("static/images/GraphTool/plotter.png", bbox_inches='tight')
    
    return render_template("Graph/graph1.html", graph = "static/images/GraphTool/plotter.png", message = "Graph plotted successfully")


@app.route("/plot_piechart", methods = ["GET","POST"])
def plot_pie():
    
    feature31 = request.form.get("feature31")
    if feature31:
        feature31 = feature31.replace(","," ")
        
    count = df[feature31].value_counts()
    plt.figure(figsize=(10,10))
    plt.pie(count, labels = count.index, autopct='%1.1f%%', shadow=True, startangle=90)
    
    plt.title(f"Pie chart for {feature31}")
    
    plt.savefig("static/images/GraphTool/pie.png", bbox_inches='tight')
    
    return render_template("Graph/graph2.html", graph = "static/images/GraphTool/pie.png", message = "Pie chart plotted successfully")

@app.route("/plot_gbarplot", methods = ["GET","POST"])
def gbarplot():
    
    target_feature = request.form.get("feature41")
    
    if target_feature:
        target_feature = target_feature.replace(","," ")
    
    features = request.form.getlist("feature42")
    features = [feature.replace(","," ") for feature in features]
    
    feature41 = features[0];
    feature42 = features[1];
    
    group_data = df.groupby(target_feature)[[feature41,feature42]].mean()
    
    plt.figure(figsize=(15,15))
    width = 0.4 # width of bar
    
    bar_positions1 = np.arange(len(group_data))
    bar_position2 = bar_positions1 + width
    
    plt.bar(bar_positions1, group_data[feature41], width=width, label=feature41)
    plt.bar(bar_position2, group_data[feature42], width=width, label=feature42)
    
    plt.xticks((bar_positions1+bar_position2)/2, group_data.index)
    plt.xlabel(target_feature)
    plt.ylabel("Average Value")
    plt.title(f"Grouped bar plot for {feature41} and {feature42}")
    plt.legend()
    
    plt.savefig("static/images/GraphTool/gbarplot.png", bbox_inches='tight')
    
    return render_template("Graph/graph3.html", graph = "static/images/GraphTool/gbarplot.png", message = "Grouped bar plot plotted successfully")








@app.route("/predict_knn_cls", methods = ["GET","POST"])
def predict_knn_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = knn_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "K Nearest Neighbors Classifier", prediction=score[0])

@app.route("/predict_svc", methods = ["GET","POST"])
def predict_svc():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = support_vector_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Support Vector Classifier", prediction=score[0])

@app.route("/predict_naive_bayes", methods = ["GET","POST"])
def predict_naive_bayes():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = native_bayes_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Naive Bayes Classifier", prediction=score[0])

@app.route("/predict_decision_tree_cls", methods = ["GET","POST"])
def predict_decision_tree_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = decision_tree_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Decision Tree Classifier", prediction=score[0])

@app.route("/predict_random_forest_cls", methods = ["GET","POST"])
def predict_random_forest_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = random_forest_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Random Forest Classifier", prediction=score[0])

@app.route("/predict_adaboost_cls", methods = ["GET","POST"])
def predict_adaboost_cls():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = adaboost_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "AdaBoost Classifier", prediction=score[0])

@app.route("/predict_gradient_boosting_cls", methods = ["GET","POST"])
def predict_gradient_boosting_cls():
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = gradientboost_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Gradient Boosting Classifier", prediction=score[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")

