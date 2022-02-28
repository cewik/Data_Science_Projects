import pandas as pd      
import numpy as np 
import streamlit as st
import sklearn
import pickle

# from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Lasso, LassoCV
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import cross_validate, cross_val_score

df = pd.read_csv("final_scout_not_dummy.csv").copy()
df_new = df[["make_model", "hp_kW", "km","age", "price", "Gearing_Type"]]
df_new.drop(index=[2614], inplace =True)
df_new = df_new[~(df_new.price>35000)]

df_new = pd.get_dummies(df_new)


X= df_new.drop(columns="price")
y= df_new.price


a = st.selectbox("make_model:",["Audi A1", "Audi A3", "Opel Astra", "Opel Corsa", "Opel Insignia", "Renault Clio", "Renault Duster", "Renault Espace"])
b = st.selectbox("Gearing_Type:", ["Manual", "Automatic", "Semi-automatic"])
c = st.slider("hp_kW:",min_value=44, max_value=294, value=0, step=1)
d = st.selectbox("age:",[0,1,2,3])
e = float(st.number_input("km:",value=0, step=10))
#st.write(f"you selected {d}")


my_dict = {"make_model":a, "Gearing_Type": b, "hp_kW": c, "age":d, "km":e }
my_dict = pd.DataFrame([my_dict])
my_dict = pd.get_dummies(my_dict)
my_dict = my_dict.reindex(columns = X.columns, fill_value=0)
#my_dict


final_scaler = pickle.load(open("final_scaler", "rb"))
my_dict = final_scaler.transform(my_dict)
#my_dict

final_model = pickle.load(open("final_model_autoscout", "rb"))
#st.button("Press me")
if st.button("Press me"):
    st.success("Analyzing")
    predictions = final_model.predict(my_dict)
    #predictions_proba = final_model.predict_proba(my_dict)
    st.write(predictions)


#price,km,age,,hp_kW,Gearing_Type,
#15770,56013.0,,3.0,,66.0,Automatic,


# Opel Corsa
# 70.0000
# 48,525.0000
# 3.0000
# 10900
# Semi-automatic