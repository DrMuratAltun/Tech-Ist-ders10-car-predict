#!/usr/bin/env python
# coding: utf-8

# # Araba Fiyatı Tahmin Eden Model ve Deployment


#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder



#Load data
df=pd.read_excel('cars.xls')







X=df.drop('Price',axis=1)
y=df[['Price']]



X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.2,
                                               random_state=42)




preproccer=ColumnTransformer(transformers=[('num',StandardScaler(),
                                           ['Mileage','Cylinder','Liter','Doors']),
                            ('cat',OneHotEncoder(),['Make','Model','Trim','Type'])])




model=LinearRegression()
pipe=Pipeline(steps=[('preprocessor',preproccer),
                    ('model',model)])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
mean_squared_error(y_test,y_pred)**0.5,r2_score(y_test,y_pred)

import streamlit as st
def price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather):
	input_data=pd.DataFrame({
		'Make':[make],
		'Model':[model],
		'Trim':[trim],
		'Mileage':[mileage],
		'Type':[car_type],
		'Car_type':[car_type],
		'Cylinder':[cylinder],
		'Liter':[liter],
		'Doors':[doors],
		'Cruise':[cruise],
		'Sound':[sound],
		'Leather':[leather]
		})
	prediction=pipe.predict(input_data)[0]
	return prediction
st.title("Araba Fiyatı Tahmin :red_car: @drmurataltun")
st.write("Arabanın özelliklerini seçin")
make=st.selectbox("Marka",df['Make'].unique())
model=st.selectbox("Model",df[df['Make']==make]['Model'].unique())
trim=st.selectbox("Trim",df[(df['Make']==make) & (df['Model']==model)]['Trim'].unique())
mileage=st.number_input("Kilometre",200,60000)
car_type=st.selectbox("Tipi",df[(df['Make']==make) & (df['Model']==model) & (df['Trim']==trim )]['Type'].unique())
cylinder=st.selectbox("Silindir",df['Cylinder'].unique())
liter=st.number_input("Liter",1,6)
doors=st.selectbox("Kapı",df['Doors'].unique())
cruise=st.radio("Hız S.",[True,False])
sound=st.radio("Ses Sistemi",[True,False])
leather=st.radio("Deri döşeme",[True,False])
if st.button("Tahmin"):
	pred=price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather)

	st.write("11062024:Predicted Price :red_car:  $",round(pred[0],2))




