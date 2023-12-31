from flask import Blueprint, render_template, request
phase1 = Blueprint('phase1', __name__, template_folder='templates', static_folder='static')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@phase1.route("/main_page", methods=["GET","POST"])
def main_page():
    global df
    from StartersPy.starter import df
    return render_template("Phase1/slot2intro.html")

#Show Dataset
@phase1.route("/show")
def show():
    return render_template("Phase1/dataset.html", dataset = df.to_html())

#Insights1
@phase1.route("/insights1")
def insights1():
    #column names
    cols = list(df.columns)
    
    # Row 1:- no. of missing values
    misses = df.isnull().sum()
    
    # Row 2:- no. of present values
    present = list(df.count())
    
    # Row 3:- type of data
    dtypes = list(df.dtypes.astype("str"))
    
    # Row 4:- no. of unique values
    unique = list(df.nunique())
    
    row1 = pd.DataFrame([present], columns=cols)
    row2 = pd.DataFrame([misses], columns=cols)
    row3 = pd.DataFrame([dtypes], columns=cols)
    row4 = pd.DataFrame([unique], columns=cols)
   
    df_req = pd.concat([row1, row2, row3,row4], keys = ["No. of Values","Missing Values","Data Type","Unique Values"])
    df_req = df_req.droplevel(1)
    return render_template("Phase1/insights1.html", dataset = df_req.to_html())

#insights2
@phase1.route("/insights2")
def insights2():
    global df_req
    cols = list(df.columns)
    
    num = df.select_dtypes(include=["int64","float64"])
    if num.empty:
        return render_template("Phase1/insights1.html", dataset = "No Numerical Features Found");   
    
    # Row 1:- minimum values
    rep = "NC"
    min_val = list(df.min())
    min_val = [rep if type(x) == str else x for x in min_val]

    # Row 2:- maximum values
    max_val = list(df.max())
    max_val = [rep if type(x) == str else x for x in max_val]

    # Calculation for further steps
    df_copy = df.copy()
    array = []
    a = list(df.dtypes.astype("str"))
    c = list(df.columns)
    for i in range(len(a)):
        if a[i] == "object":
            change = c[i]
            array.append(i)
            df_copy[change] = -1

    # Row 3:- Difference in min and max values
    diff = df_copy.max() - df_copy.min()

    # Row 4:- Mean of the values
    mean = df_copy.mean()

    # Row 5:- Median of the values
    median = df_copy.median()

    # Row 6:- Mean/Median difference
    diff2 = list(np.array(mean) - np.array(median))

    # Row 7:- Standard Deviation of the values
    std = df_copy.std()

    row1 = pd.DataFrame([min_val], columns=cols)
    row2 = pd.DataFrame([max_val], columns=cols)
    row3 = pd.DataFrame([diff], columns=cols)
    row4 = pd.DataFrame([mean], columns=cols)
    row5 = pd.DataFrame([median], columns=cols)
    row6 = pd.DataFrame([diff2], columns=cols)
    row7 = pd.DataFrame([std], columns=cols)

    df_req = pd.concat([row1, row2, row3,row4, row5, row6, row7], keys = ["Minimum","Maximum","Difference","Mean","Median","Mean-Median Difference","Standard Deviation"])
    df_req = df_req.droplevel(1)

    for i in range(len(array)):
        index = array[i]
        change = c[index]
        df_req[change] = "NC"
        
        
    # this code will fail if all are categorical
    return render_template("Phase1/insights2.html", dataset = df_req.to_html())

#round to 2 decimal
@phase1.route("/round2")
def round2():
    return render_template("Phase1/insights2.html", dataset = df_req.round(2).to_html())

#round to 3 decimal
@phase1.route("/round3")
def round3():
    return render_template("Phase1/insights2.html", dataset = df_req.round(3).to_html())

#Categorical Analysis
@phase1.route("/category")
def category():
    categorical_features = df.select_dtypes(include=["object"])
    if categorical_features.empty:
        return render_template("Phase1/category.html", dataset = "No Categorical Features")
    
    arr = categorical_features.columns.tolist()
    return render_template("Phase1/category.html", dataset = categorical_features.describe().to_html(), columns = arr)

#Visualization-I
@phase1.route("/visual1", methods = ["GET", "POST"])
def visual1():
    num = df.select_dtypes(include=["int64","float64"])
    if num.empty:
        return render_template("Phase1/visualization1.html", message1 = "No Numerical Features Found")
    plt.figure(figsize=(12,12))
    sns.heatmap(num.corr().round(2), annot=True, cmap="rainbow")
    plt.savefig("static/images/visual1/heatmap.png", bbox_inches = 'tight')
    plt.clf()
    sns.histplot(data=df,x=df.columns[-1], bins = 60, color = "r")
    plt.savefig("static/images/visual1/hist.png", bbox_inches = 'tight')
    
    
    # for histograms
    columns = list(num.columns)    
    return render_template("Phase1/visualization1.html", graph1_url = "static/images/visual1/heatmap.png", message1 = "Correlation Heatmap", graph2_url = "static/images/visual1/hist.png", message2 = "Histogram of the Target Variable",columns=columns)

# Histogram Generator
@phase1.route("/histograms", methods = ["GET", "POST"])
def histograms():
    arr = request.form.getlist('columns')
    arr = [i.replace(","," ") for i in arr]
    
    
    if(len(arr) > 9):
        return render_template("Phase1/visualization2.html", message = "Select maximum 9 features")
    if len(arr) != 0:
        sns.set_style("darkgrid")
        
        if(len(arr) == 1):
            plt.figure(figsize=(12,12))
            sns.histplot(x = df[arr[0]], bins = 30, kde = True, color = "r")
        
        elif(len(arr) == 2):
            fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (15,10))
            for i in range(len(arr)):
                if (i == 0):
                    sns.histplot(x = df[arr[0]], ax = axes[0], bins = 30, kde = True, color = "r")
                if (i == 1):
                    sns.histplot(x = df[arr[1]], ax = axes[1], bins = 30, kde = True, color = "b")
        
        elif(len(arr) > 2 and len(arr) < 5):
            fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15,15))
            for i in range(len(arr)):
                if (i == 0):
                    sns.histplot(x = df[arr[0]], ax = axes[0, 0], bins = 30, kde = True, color = "r")
                if (i == 1):
                    sns.histplot(x = df[arr[1]], ax = axes[0, 1], bins = 30, kde = True, color = "b")
                if (i == 2):
                    sns.histplot(x = df[arr[2]], ax = axes[1, 0], bins = 30, kde = True, color = "b")
                if (i == 3):
                    sns.histplot(x = df[arr[3]], ax = axes[1, 1], bins = 30, kde = True, color = "black")
        elif(len(arr) == 6):
            fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (15,10))
            for i in range(len(arr)):
                if (i == 0):
                    sns.histplot(x = df[arr[0]], ax = axes[0, 0], bins = 30, kde = True, color = "r")
                if (i == 1):
                    sns.histplot(x = df[arr[1]], ax = axes[0, 1], bins = 30, kde = True, color = "b")
                if (i == 2):
                    sns.histplot(x = df[arr[2]], ax = axes[0, 2], bins = 30, kde = True, color = "black")
                if (i == 3):
                    sns.histplot(x = df[arr[3]], ax = axes[1, 0], bins = 30, kde = True, color = "b")
                if (i == 4):
                    sns.histplot(x = df[arr[4]], ax = axes[1, 1], bins = 30, kde = True, color = "r")
                if (i == 5):
                    sns.histplot(x = df[arr[5]], ax = axes[1, 2], bins = 30, kde = True, color = "g")
        else:
            fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (15,15))
            for i in range(len(arr)):
                if (i == 0):
                    sns.histplot(x = df[arr[0]], ax = axes[0, 0], bins = 30, kde = True, color = "r")
                if (i == 1):
                    sns.histplot(x = df[arr[1]], ax = axes[0, 1], bins = 30, kde = True, color = "b")
                if (i == 2):
                    sns.histplot(x = df[arr[2]], ax = axes[0, 2], bins = 30, kde = True, color = "black")
                if (i == 3):
                    sns.histplot(x = df[arr[3]], ax = axes[1, 0], bins = 30, kde = True, color = "b")
                if (i == 4):
                    sns.histplot(x = df[arr[4]], ax = axes[1, 1], bins = 30, kde = True, color = "r")
                if (i == 5):
                    sns.histplot(x = df[arr[5]], ax = axes[1, 2], bins = 30, kde = True, color = "g")
                if (i == 6):
                    sns.histplot(x = df[arr[6]], ax = axes[2, 0], bins = 30, kde = True, color = "black")
                if (i == 7):
                    sns.histplot(x = df[arr[7]], ax = axes[2, 1], bins = 30, kde = True, color = "g")
                if (i == 8):
                    sns.histplot(x = df[arr[8]], ax = axes[2, 2], bins = 30, kde = True, color = "r")
                    
        plt.savefig("static/images/visual1.1/histograms.png", bbox_inches = 'tight') 
        return render_template("Phase1/visualization2.html", graph3_url = "static/images/visual1.1/histograms.png", message = "Histograms of the Selected Features")
    else:
        return render_template("Phase1/visualization2.html", message = "Please select atleast one feature")
        