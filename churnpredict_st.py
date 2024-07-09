import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler


model = CatBoostClassifier()
model.load_model('catboost_model2.cbm')

scaler = StandardScaler()

numeric_features=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
categorical_features=['Geography', 'Gender']


st.title("Bank Customer Churn Prediction")
st.write("""
This app predicts whether a customer will exit based on various features.
You can either input the data manually or upload a CSV file.
""")


CreditScore=st.number_input("Credit Score", min_value=0, max_value=1000, value=500)
Geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])
Gender = st.selectbox("Gender", ['Male', 'Female'])
Age = st.number_input("Age", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
Balance = st.number_input("Balance", min_value=0.0, value=1000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
HasCrCard = st.selectbox("Has Credit Card", [0, 1])
IsActiveMember = st.selectbox("Is Active Member", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)


original_input_data = {
    'CreditScore': CreditScore,
    'Geography': Geography,
    'Gender': Gender,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,
    'EstimatedSalary': EstimatedSalary
}


if st.button("Predict"):

    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [Geography],
        'Gender': [Gender],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })

    input_data = pd.get_dummies(input_data, columns=categorical_features, drop_first=True)

    expected_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                        'Geography_Germany', 'Geography_Spain', 'Gender_Male']
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[expected_columns]

    scaled_input_data = input_data.copy()
    scaled_input_data[numeric_features] = scaler.fit_transform(input_data[numeric_features])

    prediction = model.predict(scaled_input_data)[0]

    st.write("Prediction (1 means exit, 0 means stay):", int(prediction))

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    original_df = df.copy()
    
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    expected_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                        'Geography_Germany', 'Geography_Spain', 'Gender_Male']
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]

    scaled_df = df.copy()
    scaled_df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    predictions = model.predict(scaled_df)
    
    df['Prediction'] = predictions
    
    st.write("Predictions for the uploaded file:")
    st.dataframe(pd.concat([original_df, df[['Prediction']]], axis=1))
