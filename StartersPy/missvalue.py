from flask import Blueprint, render_template, request, redirect, url_for
missvalue = Blueprint('missvalue', __name__, template_folder='templates', static_folder='static')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Missing Value Analysis
@missvalue.route("/phase2")
def phase2():
    global null_df_copy,y,df
    from StartersPy.Phase1 import df 
       
    null_df = pd.DataFrame(df.isnull().sum().astype(int), columns = ["Null Count"])
    if (null_df["Null Count"].sum() == 0):
        return render_template("Phase2/missvalue.html", dataset = "No Missing Values Found")
    null_df=null_df[null_df["Null Count"] != 0]
    null_df["Null Percentage"] = (null_df["Null Count"] / len(df)) * 100
    null_df["Null Percentage"] = null_df["Null Percentage"].round(2)
    plt.clf()
    null_df["Null Count"].plot(kind="bar", title = "Bar Plot",
                           ylabel="Miss Value Count", color = "b")   
    plt.savefig("static/images/miss/miss_bar.png", bbox_inches ="tight")
    plt.clf() 
    null_df_copy = null_df.copy()
    null_df = null_df.sort_values("Null Count", ascending = False)
    message = "Your dataset has " + str(len(null_df)) + " features with missing values out of " + str(len(df.columns)) + " features."
    null_df.loc["Total"] = null_df.sum()
    
    
    # Imputation Technique through median of feature having no missing values and only few unique values
    feat_list = df.nunique().to_list()
    feat_list_idx = []
    for i in range(len(feat_list)):
        if(feat_list[i] > 1 and feat_list[i] < 15):
            feat_list_idx.append(i)
    feat_list = [df.columns.to_list()[i] for i in feat_list_idx] # Feature list having less unique values    
    
    flag = False
    feat = []
    for i in range(len(feat_list)):
        if df[null_df.T.columns.to_list()[i]].dtype == "object":
            flag = True
            feat.append(null_df.T.columns.to_list()[i])
        
        
    
    for i in feat:
        feat_list.remove(i)
        
    temp=null_df_copy.T.columns.to_list()
    y=[]
    
    for i in df.select_dtypes(include=["object"]).columns.to_list():
        if i in temp:
            y.append(i)
            
    return render_template("Phase2/missvalue.html", dataset = null_df.to_html(), message = message, bar_url = "static/images/miss/miss_bar.png", features = feat_list)

# Detecting Outliers Through Boxplots
@missvalue.route("/boxplots", methods = ["POST"])
def boxplots():
    global select_list,x
    select_list = request.form.getlist("columns") # Feature list selected by user
    select_list = [i.replace(","," ") for i in select_list]
    if(len(select_list) != 1):
        return render_template("Phase2/missvalue2.html", message="Please select exactly one feature")
    x=df.isnull().sum().to_list()
    count=0
    for i in range(len(x)):
        if(x[i]==0):
            count += 1
    x=null_df_copy.index.to_list()
    for i in range(len(x)):
        plt.figure(figsize=(15,10))
        sns.boxplot(x=df[select_list[0]], y=df[x[i]], data=df, palette = "winter")
        plt.savefig(f"static/images/miss/boxplot{i}.png", bbox_inches ="tight")
        plt.clf()
    images = [f"static/images/miss/boxplot{i}.png" for i in range(len(x))]
            
    for i in y:
        if i in x:
            x.remove(i)   
    
        
    return render_template("Phase2/missvalue2.html", length = len(x), images=images, message = "BoxPlots to see the outliers!", columns_numerical = x)
    
# Dataset Containing rows with missing values only
@missvalue.route("/show_miss")
def show_miss():
    return render_template("Phase2/miss_dataset.html", dataset = df[df.isnull().any(axis=1)].replace(np.nan, '', regex=True).to_html())

# Missing Value Imputation
@missvalue.route("/fill_misses_numerical", methods = ["POST"])
def miss_fill():
    features=request.form.getlist("columns_num")
    features = [i.replace(","," ") for i in features]

    array=list(np.unique(df[select_list[0]]))

    for i in range(len(features)):
        for j in range(len(array)):
            feature = features[i]
            target=array[j]
            median=df[df[select_list[0]]==target][feature].median()
            df[feature].fillna(median,inplace=True)
    plt.clf()
    return redirect(url_for("missvalue.phase2"))

@missvalue.route("/fill_misses_categorical", methods = ["GET","POST"])
def fill_misses_categorical():
    features = y
    for i in features:
    
        mode_value = df[i].mode().iloc[0]
        df[i] = df[i].fillna(mode_value)

        
    return redirect(url_for("missvalue.phase2"))