from flask import Flask, request, render_template
import pickle
import pandas as pd 
import numpy as np
import sklearn

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
# from sklearn.model_selection import cross_val_score, cross_validate

app = Flask(__name__)

@app.route("/", methods = ["GET","POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        a = request.form.get("make_model")
        b = request.form.get("Gearing_Type")
        c = float(request.form.get("hp_kW"))
        d = int(request.form.get("age"))
        e = int(request.form.get("km"))
        
 
        df = pd.read_csv("final_scout_not_dummy.csv").copy()
        df_new = df[["make_model", "hp_kW", "km","age", "price", "Gearing_Type"]]
        df_new.drop(index=[2614], inplace =True)
        df_new = df_new[~(df_new.price>35000)]
        df_new = pd.get_dummies(df_new)


        X= df_new.drop(columns="price")
        y= df_new.price


        my_dict = {"make_model":a, "Gearing_Type": b, "hp_kW": c, "age":d, "km":e }
        my_dict = pd.DataFrame([my_dict])
        # my_dict.age.astype(int) # [16781.45983364]
        my_dict = pd.get_dummies(my_dict)
        my_dict = my_dict.reindex(columns = X.columns, fill_value=0)


        final_scaler = pickle.load(open("final_scaler", "rb"))
        my_dict = final_scaler.transform(my_dict)
        

        final_model = pickle.load(open("final_model_autoscout", "rb"))
        predictions = final_model.predict(my_dict)
   


        return render_template("index.html",result = f"predicton: $ {predictions[0]:.2f}") 




if __name__ == "__main__":
    app.run(debug=True, host = "0.0.0.0")