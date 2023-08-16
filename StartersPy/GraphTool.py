from flask import Blueprint, render_template, request, jsonify
GraphTool = Blueprint('GraphTool', __name__, template_folder='templates', static_folder='static')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@GraphTool.route("/graph", methods = ["GET","POST"])
def grapher():

    global send,df
    from StartersPy.Encoding import df
    
    columns = df.columns.to_list()
    return render_template("/Graph/main.html",columns=columns)

@GraphTool.route("/plot_graph", methods = ["GET","POST"])
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


@GraphTool.route("/plot_piechart", methods = ["GET","POST"])
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

@GraphTool.route("/plot_gbarplot", methods = ["GET","POST"])
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


