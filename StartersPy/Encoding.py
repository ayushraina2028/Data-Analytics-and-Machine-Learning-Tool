from flask import Blueprint, render_template, request, redirect, url_for
encoder = Blueprint('encoder', __name__, template_folder='templates', static_folder='static')

#Encoding Categorical Features
@encoder.route("/phase3")
def phase3():
    
    global send,df
    from StartersPy.missvalue import df

    x=df.dtypes.astype(str).to_list()
    count = 0
    idx=[] #idx of categorical features
    unique_features=[]
    for i in range(len(x)):
        if x[i]=="object":
            count+=1;
            idx.append(i)
            unique_features.append(list(df[df.columns.to_list()[i]].unique()))
    if count==0:
        return render_template("Encoding/Encoding1.html", message1="No Categorical Features Found",message2="Your can proceed to next step")
    feature_names=[df.columns.to_list()[i] for i in idx] #categorical feature names
    send={} # dictionary of categorical features and their unique values
    for i in range(len(feature_names)):
        send[feature_names[i]]=unique_features[i]
    return render_template("Encoding/Encoding.html", message1="Your dataset has "+str(count)+" categorical features",message2="Encoding them to Numeric Values here"
                           ,send=send)

@encoder.route("/encoding",methods=["GET","POST"])
def encode():
    global encodings,feature
    encoded_values=[] #list of encoded values
    array=[] #list of categorical features
    feature=[] #list of categorical features
    for features,values in send.items():
        array.append(send[features])
        encoded_values.append([request.form.get(f"{value}") for value in values])
        
    encoded_values=[[int(x) if x is not None else None for x in sub_list] for sub_list in encoded_values]
    child_list=[] #list of index and encoded values
    x=[sublist for sublist in encoded_values if None not in sublist]
    child_list.append([encoded_values.index(x[0]),x[0]])
    encodings={} #dictionary of encoded values
    for i in range(len(child_list[0][1])):
        encodings[array[child_list[0][0]][i]]=child_list[0][1][i]
    for features in send.keys():
        feature.append(features)
    feature=feature[child_list[0][0]]
    
    return render_template("Encoding/Encoding2.html",encodings=encodings)

@encoder.route("/encode_it",methods=["GET","POST"])
def encode_1():
    global df1
    df[feature]=[encodings[x] for x in df[feature]]
    df1= df.copy()
    return redirect(url_for("Encoding.phase3"))