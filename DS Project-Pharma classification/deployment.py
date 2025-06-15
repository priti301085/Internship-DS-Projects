# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:09:45 2024

@author: Lenovo
"""


import pickle
import streamlit as st

# Load model
load = open('xgb.pkl', 'rb')
model = pickle.load(load)

# Dictionary mapping of disease indices to names

disease_names = {
    model.classes_[0]: "Cirrhosis",
    model.classes_[1]: "Fibrosis",
    model.classes_[2]: "Hepatitis",
    model.classes_[3]: "No Disease",
    model.classes_[4]: "Suspect Disease"
}
# Prediction function
def predict(age, sex, albumin, alkaline_phosphatase, alanine_aminotransferase, aspartate_aminotransferase,
            bilirubin, cholinesterase, cholesterol, creatinina, gamma_glutamyl, protein):
    # Make prediction and get probability
    prediction = model.predict([[age, sex, albumin, alkaline_phosphatase, alanine_aminotransferase,
                                 aspartate_aminotransferase, bilirubin, cholinesterase, cholesterol,
                                 creatinina, gamma_glutamyl, protein]])
    
    probability = model.predict_proba([[age, sex, albumin, alkaline_phosphatase, alanine_aminotransferase,
                                        aspartate_aminotransferase, bilirubin, cholinesterase, cholesterol,
                                        creatinina, gamma_glutamyl, protein]])
    
    predicted_index = prediction[0]
    predicted_disease = disease_names.get(predicted_index, "Unknown Disease")
    predicted_probability = probability[0][predicted_index] * 100  # Get probability for predicted class
    
    return predicted_disease, predicted_probability

def main():
    st.title('Liver Disease Prediction')

    # User inputs
    age = st.number_input('Age: ', min_value=0, step=1)
    sex = st.radio('Sex:', [0, 1], format_func=lambda x: 'Male' if x == 0 else 'Female')
    albumin = st.number_input('Albumin: ', min_value=0.0, step=0.01)
    alkaline_phosphatase = st.number_input('Alkaline Phosphatase: ', min_value=0.0, step=0.01)
    alanine_aminotransferase = st.number_input('Alanine Aminotransferase: ', min_value=0.0, step=0.01)
    aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase: ', min_value=0.0, step=0.01)
    bilirubin = st.number_input('Bilirubin: ', min_value=0.0, step=0.01)
    cholinesterase = st.number_input('Cholinesterase: ', min_value=0.0, step=0.01)
    cholesterol = st.number_input('Cholesterol: ', min_value=0.0, step=0.01)
    creatinina = st.number_input('Creatinina: ', min_value=0.0, step=0.01)
    gamma_glutamyl = st.number_input('Gamma Glutamyl : ', min_value=0.0, step=0.01)
    protein = st.number_input('Protein: ', min_value=0.0, step=0.01)

    if st.button('Predict'):
        # Make prediction and display result
        disease, probability = predict(age, sex, albumin, alkaline_phosphatase, alanine_aminotransferase, 
                                       aspartate_aminotransferase, bilirubin, cholinesterase, cholesterol, 
                                       creatinina, gamma_glutamyl, protein)
        
        st.subheader("Predicted Disease:")
        st.write(f"{disease} with a probability of {probability:.2f}%")

# Run main function
if __name__ == '__main__':
    main()
