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