from flask import Blueprint, render_template, request, jsonify
NaiveBayesClassification = Blueprint('NaiveBayesClassification', __name__, template_folder='templates', static_folder='static')

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
        
@NaiveBayesClassification.route("/train_naive_bayes_classifier", methods = ["GET","POST"])
def train_native_bayes_classifier():
    global native_bayes_classifier
    
    global X_train,X_test,y_train,y_test,target,training
    from ModelsPy.modelPage import X_train,X_test,y_train,y_test,target,training
    
    
    classify = request.form.get("algos")
    native_bayes_classifier=naive_bayes_classifier(X_train,y_train,types=classify)
    return render_template("models/NaiveBayes/NaiveBayes.html",
                            target=target, trains=training,train_status=f"{classify} Naive Bayes Model is trained Successfully",
                            columns=training,model="naive_bayes")    

@NaiveBayesClassification.route("/test_naive_bayes_classifier", methods = ["GET","POST"])
def test_native_bayes_classifier():
    
        from ModelsPy.Accuracies import check_accuracy
        
        score=check_accuracy(y_test,native_bayes_classifier.predict(X_test))
        score=score*100
        return jsonify({"score":score})

@NaiveBayesClassification.route("/predict_naive_bayes", methods = ["GET","POST"])
def predict_naive_bayes():
    
    data = request.form.getlist("data")
    data = [float(d) for d in data]
    score = native_bayes_classifier.predict([data])
    return render_template("Prediction/prediction.html", modelname = "Naive Bayes Classifier", prediction=score[0])


