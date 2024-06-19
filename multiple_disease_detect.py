# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:52:45 2024

@author: HP
"""

import pickle
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

#loading the saved models

with open('Thyroid_model.sav', 'rb') as file:
    Thyroid_model = pickle.load(file)

with open("Lung_cancer_model.sav", 'rb') as file:
    Lung_Cancer_model = pickle.load(file)

with open('Breast_cancer_model.sav', 'rb') as file:
    Breast_Cancer_model = pickle.load(file)
    
with open('Diabetes_model.sav', 'rb') as file:
    Diabetes_model = pickle.load(file)
    
with open("Asthma_model.sav", 'rb') as file:
    Asthma_model = pickle.load(file)

with open("PCOS_data.sav", 'rb') as file:
    PCOS_model = pickle.load(file)
    
with open("Heart_stroke.sav", 'rb') as file:
    Heart_stroke_model = pickle.load(file)
        
with open("Heart_Disease_model.sav", 'rb') as file:
    Heart_Disease_model = pickle.load(file)
    
with open('parkinson_model.sav', 'rb') as file:
    parkinson_model = pickle.load(file)
    
with open("Migraine_model.sav", 'rb') as file:
    Migraine_model = pickle.load(file)
    
with open("Covid-19.sav",'rb') as file:
    Covid_19 = pickle.load(file)
    
with open("Alzheimer_model.sav", 'rb') as file:
    Alzheimer_model = pickle.load(file)
    
with open("depression_anxiety_data.sav",'rb') as file:
    depression_anxiety_data = pickle.load(file)
    
with open("kidney disease_model.sav", 'rb') as file:
    kidney_disease_model = pickle.load(file)
    
# List of diseases
diseases = [
    'Thyroid', 'Covid-19', 'Lung Cancer', 'Depression & Anxiety',
    'Breast Cancer', 'Diabetes', 'Asthma', 'PCOS', 'Heart Stroke',
    'Migraine', 'Cardiovascular Disease', 'Parkinson', 'Alzheimer', 'Kidney Disease'
]

# Disease to model mapping
disease_models = {
    'Thyroid': Thyroid_model,
    'Covid-19': Covid_19,
    'Lung Cancer': Lung_Cancer_model,
    'Depression & Anxiety': depression_anxiety_data,
    'Breast Cancer': Breast_Cancer_model,
    'Diabetes': Diabetes_model,
    'Asthma': Asthma_model,
    'PCOS': PCOS_model,
    'Heart Stroke': Heart_stroke_model,
    'Migraine': Migraine_model,
    'Cardiovascular Disease': Heart_Disease_model,
    'Parkinson': parkinson_model,
    'Alzheimer': Alzheimer_model,
    'Kidney Disease': kidney_disease_model
}

# Function to filter diseases based on search query
def filter_diseases(query):
    return [disease for disease in diseases if query.lower() in disease.lower()]

st.markdown("""
    <h1 style='text-align: center; color: #1E999F; font-size: 95px;'>MedScan</h1>
    """, unsafe_allow_html=True)
    
# Sidebar for navigation
with st.sidebar:
    # Search bar for diseases
    search_query = st.text_input("Search Disease")
    
    # Filter diseases based on search query
    filtered_diseases = filter_diseases(search_query)
    
    # Disease selection menu
    selected = option_menu('Multiple Disease Detection System',
                           filtered_diseases,
                           icons = ['capsule', 
                                    'virus',
                                    'lungs', 
                                    'peace',
                                    'postcard-heart', 
                                    'prescription2', 
                                    'clipboard2-pulse',  
                                    'gender-female', 
                                    'heart-pulse',
                                    'eyeglasses',
                                    'heart',
                                    'lightning-charge',
                                    'bandaid',
                                    'file-medical-fill'],
                           default_index = 0) 

# Display selected disease
st.write(f"You selected: {selected}")

# Add the rest of your disease detection system code below, using the selected model
# Example: Use the selected model to make predictions
model = disease_models[selected]

   
if selected == 'Thyroid':
    # Page title
    st.title('Thyroid Detection')
    
    # Input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=0, help="Enter your age in years.")
    with col2:
        sex = st.selectbox('Gender', ['male', 'female'], help="Select your gender.")
    with col3:
        on_thyroxine = st.selectbox('Thyroxine', ['yes', 'no'], help="Are you currently on thyroxine medication?")
    with col1:
        on_antithyroid_meds = st.selectbox('On antithyroid medicines', ['yes', 'no'], help="Are you currently on antithyroid medications?")
    with col2:
        sick = st.selectbox('Sick', ['yes', 'no'], help="Are you currently sick?")
    with col3:
        pregnant = st.selectbox('Pregnant', ['yes', 'no'], help="Are you pregnant?")
    with col1:
        thyroid_surgery = st.selectbox('Thyroid surgery', ['yes', 'no'], help="Have you had thyroid surgery?")
    with col2:
        I131_treatment = st.selectbox('I131 treatment', ['yes', 'no'], help="Have you received I131 treatment?")
    with col3:
        query_hypothyroid = st.selectbox('Query hypothyroid', ['yes', 'no'], help="Is there a query about hypothyroidism?")
    with col1:
        query_hyperthyroid = st.selectbox('Query hyperthyroid', ['yes', 'no'], help="Is there a query about hyperthyroidism?")
    with col2:
        lithium = st.selectbox('Lithium', ['yes', 'no'], help="Are you on lithium medication?")
    with col3:
        goitre = st.selectbox('Goitre', ['yes', 'no'], help="Do you have goitre?")
    with col1:
        tumor = st.selectbox('Tumor', ['yes', 'no'], help="Do you have a tumor?")
    with col2:
        hypopituitary = st.selectbox('Hypopituitary', ['yes', 'no'], help="Do you have hypopituitary condition?")
    with col3:
        psych = st.selectbox('Psych', ['yes', 'no'], help="Do you have any psychiatric condition?")
    with col1:
        TSH = st.number_input('TSH', value=0.0, help="Enter your TSH (Thyroid-stimulating hormone) level.")
    with col2:
        T3 = st.number_input('T3', value=0.0, help="Enter your T3 (Triiodothyronine) level.")
    with col3:
        TT4 = st.number_input('TT4', value=0.0, help="Enter your TT4 (Total thyroxine) level.")
    with col1:
        T4U = st.number_input('T4U', value=0.0, help="Enter your T4U (Thyroxine uptake) level.")
    with col2:
        FTI = st.number_input('FTI', value=0.0, help="Enter your FTI (Free thyroxine index).")
    with col3:
        TBG = st.number_input('TBG', value=0.0, help="Enter your TBG (Thyroxine-binding globulin) level.")

    # Convert categorical inputs to numerical if necessary
    sex = 1 if sex == 'male' else 0
    on_thyroxine = 1 if on_thyroxine == 'yes' else 0
    on_antithyroid_meds = 1 if on_antithyroid_meds == 'yes' else 0
    sick = 1 if sick == 'yes' else 0
    pregnant = 1 if pregnant == 'yes' else 0
    thyroid_surgery = 1 if thyroid_surgery == 'yes' else 0
    I131_treatment = 1 if I131_treatment == 'yes' else 0
    query_hypothyroid = 1 if query_hypothyroid == 'yes' else 0
    query_hyperthyroid = 1 if query_hyperthyroid == 'yes' else 0
    lithium = 1 if lithium == 'yes' else 0
    goitre = 1 if goitre == 'yes' else 0
    tumor = 1 if tumor == 'yes' else 0
    hypopituitary = 1 if hypopituitary == 'yes' else 0
    psych = 1 if psych == 'yes' else 0
    
    #code for Prediction
    
    #mapping from integer to thyroid condition labels
    int_to_label = {
        0: '-', 1: 'A', 2: 'AK', 3: 'B', 4: 'C', 5: 'C|I', 6: 'D', 7: 'D|R', 8: 'E',
        9: 'F', 10: 'FK', 11: 'G', 12: 'GI', 13: 'GK', 14: 'GKJ', 15: 'H|K', 16: 'I',
        17: 'J', 18: 'K', 19: 'KJ', 20: 'L', 21: 'LJ', 22: 'M', 23: 'MI', 24: 'MK',
        25: 'N', 26: 'O', 27: 'OI', 28: 'P', 29: 'Q', 30: 'R', 31: 'S', 32: 'T'
    }
    
    #mapping from labels to descriptive names
    label_to_name = {
        '-': 'No thyroid condition',
        'A': 'hyperthyroid',
        'B': 'T3 toxic',
        'C': 'toxic goitre',
        'C|I': 'toxic goitre with increased binding protein',
        'D': 'secondary toxic',
        'D|R': 'secondary toxic with discordant assay results',
        'E': 'hypothyroid',
        'F': 'primary hypothyroid',
        'FK': 'primary hypothyroid with concurrent non-thyroidal illness',
        'G': 'compensated hypothyroid',
        'GI': 'compensated hypothyroid with increased binding protein',
        'GK': 'compensated hypothyroid with concurrent non-thyroidal illness',
        'GKJ': 'compensated hypothyroid with concurrent non-thyroidal illness and decreased binding protein',
        'H|K': 'secondary hypothyroid with concurrent non-thyroidal illness',
        'I': 'increased binding protein',
        'J': 'decreased binding protein',
        'K': 'concurrent non-thyroidal illness',
        'KJ': 'concurrent non-thyroidal illness with decreased binding protein',
        'L': 'consistent with replacement therapy',
        'LJ': 'consistent with replacement therapy with decreased binding protein',
        'M': 'underreplaced',
        'MI': 'underreplaced with increased binding protein',
        'MK': 'underreplaced with concurrent non-thyroidal illness',
        'N': 'overreplaced',
        'O': 'antithyroid drugs',
        'OI': 'antithyroid drugs with increased binding protein',
        'P': 'I131 treatment',
        'Q': 'surgery',
        'R': 'discordant assay results',
        'S': 'elevated TBG',
        'T': 'elevated thyroid hormones',
        'AK': 'hyperthyroid with concurrent non-thyroidal illness',
        'LJ': 'consistent with replacement therapy with decreased binding protein',
        'C|I': 'toxic goitre with increased binding protein',
        'H|K': 'secondary hypothyroid with concurrent non-thyroidal illness',
        'GKJ': 'compensated hypothyroid with concurrent non-thyroidal illness and decreased binding protein'
    }
    
    # Button for prediction
    if st.button('Thyroid Test Result'):
        # Create input array
        input_data = (age, sex, on_thyroxine, on_antithyroid_meds, sick, 
                      pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, 
                      query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, 
                      TSH, T3, TT4, T4U, FTI, TBG)

        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        thyroid_prediction = Thyroid_model.predict(input_data_reshaped)
        label = int_to_label[thyroid_prediction[0]]
        condition_name = label_to_name[label]
        
        # Display result
        st.success(f'Result: {condition_name}')
    
if selected == 'Lung Cancer':
    # Page title
    st.title('Lung Cancer Prediction')
    st.caption(':blue[(Note: rate values in 1-10)]')
    
    # Input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', min_value=0, max_value=120, value=0, help="Enter your age in years.")
    with col2:
        gender = st.selectbox('Gender', ['male', 'female'], help="Select your gender.")
    with col3:
        air_pollution = st.number_input('Air Pollution', min_value=0, max_value=10, value=0, help="Rate your exposure to air pollution (0-10).")
    with col1:
        alcohol_use = st.number_input('Alcohol Use', min_value=0, max_value=10, value=0, help="Rate your alcohol use (0-10).")
    with col2:
        dust_allergy = st.number_input('Dust Allergy', min_value=0, max_value=10, value=0, help="Rate the severity of your dust allergy (0-10).")
    with col3:
        occupational_hazards = st.number_input('Occupational Hazards', min_value=0, max_value=10, value=0, help="Rate your exposure to occupational hazards (0-10).")
    with col1: 
        genetic_risk = st.number_input('Genetic Risk', min_value=0, max_value=10, value=0, help="Rate your genetic risk for lung cancer (0-10).")
    with col2: 
        chronic_lung_disease = st.number_input('Chronic Lung Disease', min_value=0, max_value=10, value=0, help="Rate the severity of any chronic lung disease (0-10).")
    with col3: 
        balanced_diet = st.number_input('Balanced Diet', min_value=0, max_value=10, value=0, help="Rate how balanced your diet is (0-10).")
    with col1: 
        obesity = st.number_input('Obesity', min_value=0, max_value=10, value=0, help="Rate your level of obesity (0-10).")
    with col2: 
        smoking = st.number_input('Smoking', min_value=0, max_value=10, value=0, help="Rate your smoking habits (0-10).")
    with col3: 
        passive_smoker = st.number_input('Passive Smoker', min_value=0, max_value=10, value=0, help="Rate your exposure to secondhand smoke (0-10).")
    with col1: 
        chest_pain = st.number_input('Chest Pain', min_value=0, max_value=10, value=0, help="Rate the severity of your chest pain (0-10).")
    with col2: 
        coughing_of_blood = st.number_input('Coughing of Blood', min_value=0, max_value=10, value=0, help="Rate the frequency of coughing up blood (0-10).")
    with col3: 
        fatigue = st.number_input('Fatigue', min_value=0, max_value=10, value=0, help="Rate your level of fatigue (0-10).")
    with col1: 
        weight_loss = st.number_input('Weight Loss', min_value=0, max_value=10, value=0, help="Rate the amount of weight loss (0-10).")
    with col2: 
        shortness_of_breath = st.number_input('Shortness of Breath', min_value=0, max_value=10, value=0, help="Rate your shortness of breath (0-10).")
    with col3: 
        wheezing = st.number_input('Wheezing', min_value=0, max_value=10, value=0, help="Rate the severity of your wheezing (0-10).")
    with col1: 
        swallowing_difficulty = st.number_input('Swallowing Difficulty', min_value=0, max_value=10, value=0, help="Rate the difficulty of swallowing (0-10).")
    with col2: 
        clubbing_of_finger_nails = st.number_input('Clubbing of Finger Nails', min_value=0, max_value=10, value=0, help="Rate the severity of finger nail clubbing (0-10).")
    with col3: 
        frequent_cold = st.number_input('Frequent Cold', min_value=0, max_value=10, value=0, help="Rate the frequency of colds (0-10).")
    with col1: 
        dry_cough = st.number_input('Dry Cough', min_value=0, max_value=10, value=0, help="Rate the severity of your dry cough (0-10).")
    with col2: 
        snoring = st.number_input('Snoring', min_value=0, max_value=10, value=0, help="Rate the severity of your snoring (0-10).")
    
    gender = 1 if gender == 'female' else 2
    #button
    if st.button('Lung Cancer Result'):
        
        input_data = ([age, gender, air_pollution, alcohol_use, dust_allergy,
                       occupational_hazards, genetic_risk, chronic_lung_disease,
                       balanced_diet, obesity, smoking, passive_smoker, chest_pain,
                       coughing_of_blood, fatigue, weight_loss, shortness_of_breath,
                       wheezing, swallowing_difficulty, clubbing_of_finger_nails,
                       frequent_cold, dry_cough, snoring])
        
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        lung_cancer_prediction = Lung_Cancer_model.predict(input_data_reshaped)
    
        st.success(f'Lung cancer Level: {lung_cancer_prediction}')
    
if selected == 'Breast Cancer':
    # Page title
    st.title('Breast Cancer Detection')
    
    col1, col2, col3 = st.columns(3)
    
    # Input data
    with col1: 
        radius_mean = st.number_input('Radius mean', help="Mean of distances from center to points on the perimeter")
    with col2: 
        texture_mean = st.number_input('Texture mean', help="Standard deviation of gray-scale values")
    with col3: 
        perimeter_mean = st.number_input('Perimeter mean', help="Mean size of the core tumor perimeter")
    with col1:
        area_mean = st.number_input('Area mean', help="Mean area of the core tumor")
    with col2:
        smoothness_mean = st.number_input('Smoothness mean', help="Mean of local variation in radius lengths")
    with col3:
        compactness_mean = st.number_input('Compactness mean', help="Mean of perimeter^2 / area - 1.0")
    with col1:
        concavity_mean = st.number_input('Concavity mean', help="Mean of severity of concave portions of the contour")
    with col2:
        concave_points_mean = st.number_input('Concave points mean', help="Mean number of concave portions of the contour")
    with col3:
        symmetry_mean = st.number_input('Symmetry mean', help="Mean of symmetry")
    with col1:
        fractal_dimension_mean = st.number_input('Fractal dimension mean', help="Mean of 'coastline approximation' - 1")
    with col2:
        radius_se = st.number_input('Radius se', help="Standard error for the radius")
    with col3:
        texture_se = st.number_input('Texture se', help="Standard error for texture")
    with col1:
        perimeter_se  = st.number_input('Perimeter se', help="Standard error for perimeter")
    with col2:
        area_se = st.number_input('Area se', help="Standard error for area")
    with col3:
        smoothness_se = st.number_input('Smoothness se', help="Standard error for smoothness")
    with col1:
        compactness_se = st.number_input('Compactness se', help="Standard error for compactness")
    with col2:
        concavity_se = st.number_input('Concavity se', help="Standard error for concavity")
    with col3:
        concave_points_se = st.number_input('Concave points se', help="Standard error for concave points")
    with col1:
        symmetry_se = st.number_input('Symmetry se', help="Standard error for symmetry")
    with col2:
        fractal_dimension_se = st.number_input('Fractal dimension se', help="Standard error for fractal dimension")
    with col3:
        radius_worst = st.number_input('Radius worst', help="Worst (mean of the three largest values) radius")
    with col1:
        texture_worst = st.number_input('Texture worst', help="Worst texture")
    with col2:
        perimeter_worst = st.number_input('Perimeter worst', help="Worst perimeter")
    with col3:
        area_worst = st.number_input('Area worst', help="Worst area")
    with col1:
        smoothness_worst = st.number_input('Smoothness worst', help="Worst smoothness")
    with col2:
        compactness_worst = st.number_input('Compactness worst', help="Worst compactness")
    with col3:
        concavity_worst = st.number_input('Concavity worst', help="Worst concavity")
    with col1:
        concave_points_worst = st.number_input('Concave points worst', help="Worst concave points")
    with col2:
        symmetry_worst = st.number_input('Symmetry worst', help="Worst symmetry")
    with col3:
        fractal_dimension_worst = st.number_input('Fractal dimension worst', help="Worst fractal dimension")
     
    int_to_label = {0: 'B',1: 'M'}
    label_to_name = {'B': 'Benign', 'M': 'Maligant'}
    
    if st.button('Breast Cancer Result'):
        
        input_data = ([radius_mean, texture_mean, perimeter_mean,
        area_mean, smoothness_mean, compactness_mean, concavity_mean,
        concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se,
        fractal_dimension_se, radius_worst, texture_worst,
        perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst,
        symmetry_worst, fractal_dimension_worst])
    
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        breast_cancer_prediction = Breast_Cancer_model.predict(input_data_reshaped)
        label = int_to_label[breast_cancer_prediction[0]]
        condition_name = label_to_name[label]
        
        if breast_cancer_prediction == [0]:    
            st.success(f'No breast cancer detected as the type is {condition_name}')
        else:
            st.success(f'Breast cancer detected as the type is {condition_name}')
    
if selected == 'Diabetes':
    st.title('Diabetes Detection')
    
    col1, col2 = st.columns(2)
    
    with col1:
        Pregnancies = st.number_input('Pregnancies', help="Number of times pregnant")          
    with col2:
        Glucose = st.number_input('Glucose', help="Plasma glucose concentration (mg/dL)")                    
    with col1:
        BloodPressure = st.number_input('Blood Pressure', help="Diastolic blood pressure (mm Hg)")
    with col2:
        SkinThickness = st.number_input('Skin Thickness', help="Triceps skin fold thickness (mm)")
    with col1:
        Insulin = st.number_input('Insulin', help="2-Hour serum insulin (mu U/ml)")                       
    with col2:
        BMI = st.number_input('BMI', help="Body Mass Index (weight in kg/(height in m)^2)")                         
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', help="Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)")    
    with col2: 
        Age = st.number_input('Age', help="Age (years)")

    if st.button('Diabetes Result'):
        input_data = ([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        diabetes_prediction = Diabetes_model.predict(input_data_reshaped)                   
        
        if diabetes_prediction == [1]:
            st.success(f'Diabetes Detected')
        else:
            st.success(f'No Diabetes Detected')
            
#Display Prediction Page
if selected == 'Asthma':
    # Page title
    st.title('Asthma Prediction')
    
    # Input fields
    Age = st.number_input('Age', help="Enter your age in years")                
    Gender = st.selectbox('Gender', ['male', 'female'], help="Select your gender")               
    Smoking_Status = st.selectbox('Smoking status', ['Ex-smoker', 'Current-smoker', 'Non-smoker'], help="Select your smoking status")            
    Medication = st.selectbox('Medication', ['Controller Medication', 'Inhaler'], help="Select the type of medication you are using")   
    Peak_Flow = st.number_input('Peak flow', help="Enter your peak flow reading (L/min)")
    
    Gender = 0 if Gender == 'female' else 1
    
    if Smoking_Status == 'Ex-smoker':
        Smoking_Status = 1
    elif Smoking_Status == 'Current-smoker':
        Smoking_Status = 0
    else:
        Smoking_Status = 2
        
    Medication = 1 if Medication == 'Inhaler' else 0
    
    if st.button('Asthma result'):
        input_data=([Age, Gender, Smoking_Status, Medication, Peak_Flow])
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        prediction = Asthma_model.predict(input_data_reshaped)
        if prediction == [1]:
          st.success(f'Asthma Predicted')
        else:
          st.success(f'No Asthma Predicted')
              
if selected == 'PCOS':
    st.title('PCOS Prediction')
    
    # Input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Age  = st.number_input('Age', help="Enter your age in years")
    with col2:
        Weight = st.number_input('Weight', help="Enter your weight in kilograms")
    with col3:
        BMI = st.number_input('BMI', help="Enter your Body Mass Index (weight in kg/(height in m)^2)")          
    with col1:                      
        Hb = st.number_input('Hb', help="Enter your hemoglobin level (g/dL)")
    with col2:
        FSH = st.number_input('FSH', help="Enter your Follicle-Stimulating Hormone level (mIU/mL)")   
    with col3:         
        LH = st.number_input('LH', help="Enter your Luteinizing Hormone level (mIU/mL)")    
    with col1:        
        FSH_LH = st.number_input('FSH_LH', help="Enter the FSH to LH ratio")  
    with col2:              
        TSH = st.number_input('TSH', help="Enter your Thyroid-Stimulating Hormone level (mIU/L)")       
    with col3:          
        AMH = st.number_input('AMH', help="Enter your Anti-Müllerian Hormone level (ng/mL)")  
    with col1:    
        PRL = st.number_input('PRL', help="Enter your prolactin level (ng/mL)")    
    with col2:
        Vit_D3 = st.number_input('Vit D3', help="Enter your Vitamin D3 level (ng/mL)") 
    with col3:     
        PRG = st.number_input('PRG', help="Enter your progesterone level (ng/mL)")
    with col1:
        Follicle_No_L = st.number_input('Follicle No(L)', help="Enter the number of follicles in the left ovary")
    with col2:
        Follicle_No_R = st.number_input('Follicle No(R)', help="Enter the number of follicles in the right ovary")
    with col3:
        Avg_F_size_L = st.number_input('Avg. F_size(L)', help="Enter the average follicle size in the left ovary (mm)")
    with col1:
        Avg_F_size_R = st.number_input('Avg. F_size(R)', help="Enter the average follicle size in the right ovary (mm)")
    with col2:
        Endometrium = st.number_input('Endometrium', help="Enter the endometrium thickness (mm)")
        
    if st.button('PCOS Result'):
        input_data = ([Age, Weight, BMI, Hb, FSH, LH, FSH_LH, TSH, AMH, PRL, Vit_D3, PRG, Follicle_No_L,
                        Follicle_No_R, Avg_F_size_L, Avg_F_size_R, Endometrium])
            
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        pcos_prediction = PCOS_model.predict(input_data_reshaped)
        if pcos_prediction == [1]:
          st.success(f'PCOS Detected')
        else:
          st.success(f"No PCOS Detected")
         
if selected == 'Heart Stroke':
    st.title('Heart Stroke Prediction')
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'], help="Select your gender")                   
    with col2:
        age = st.number_input('Age', help="Enter your age in years")
    with col1:
        hypertension = st.selectbox('Hyper Tension', ['Yes', 'No'], help="Do you have hypertension?")
    with col2:
        heart_disease = st.selectbox('Heart Disease', ['Yes', 'No'], help="Do you have heart disease?")
    with col1:
        ever_married = st.selectbox('Ever Married', ['Yes', 'No'], help="Have you ever been married?")
    with col2:
        work_type = st.selectbox('Work type', ['Private', 'Self-employed', 'Govt job', 'Children', 'Never worked'], help="Select your work type")            
    with col1:
        Residence_type = st.selectbox('Residence type', ['Urban', 'Rural'], help="Select your residence type")  
    with col2:
        avg_glucose_level = st.number_input('Average glucose level', help="Enter your average glucose level (mg/dL)") 
    with col1:
        bmi = st.number_input('BMI', help="Enter your Body Mass Index (weight in kg/(height in m)^2)")         
    with col2:
        smoking_status = st.selectbox('Smoking status', ['Formerly smoked', 'Never smoked', 'Smokes', 'Unknown'], help="Select your smoking status")
    
    gender = 0 if gender == 'female' else 1
    hypertension = 0 if hypertension == 'No' else 1
    heart_disease = 0 if heart_disease == 'No' else 1
    ever_married = 0 if ever_married == 'No' else 1
    Residence_type = 0 if Residence_type == 'Rural' else 1
    if work_type == 'children': 
        work_type = 0
    elif work_type == 'Govt job':
        work_type = 1
    elif work_type == 'Never worked':
        work_type = 2
    elif work_type == 'Private':
        work_type = 3   
    else:
        work_type = 4
    
    if smoking_status == 'formerly smoked': 
        smoking_status = 0
    elif smoking_status == 'never smoked':
        smoking_status = 1
    elif smoking_status == 'smokes':
        smoking_status = 2
    else:
        smoking_status = 3
        
    if st.button('Predict Heart Stroke'):
        input_data = ([gender, age, hypertension, heart_disease, ever_married, 
                       work_type, Residence_type, avg_glucose_level, bmi, smoking_status])
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        heart_stroke_prediction = Heart_stroke_model.predict(input_data_reshaped)
        if heart_stroke_prediction == [1]:
          st.success(f"Heart Stroke predicted")
        else:
          st.success(f" No Headrt Stroke predicted")
          
if selected == 'Cardiovascular Disease':
    st.title('Cardiovascular Disease Prediction')
    
    # Display information for users
    st.write(":blue_book: Please fill in the following details for heart disease prediction.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        General_Health = st.selectbox('General Health', ['Poor', 'Very Good', 'Good', 'Fair', 'Excellent'],
                                     help="Select your general health condition.")
    with col2:
        Checkup = st.selectbox('Checkup', ['Within the past 2 years', 'Within the past year',
                                           '5 or more years ago', 'Within the past 5 years', 'Never'],
                               help="When did you last have a medical checkup?")
    with col3:
        Exercise = st.selectbox('Exercise', ['Yes', 'No'],
                                help="Do you regularly engage in physical exercise?")
    with col1:
        Skin_Cancer = st.selectbox('Skin Cancer', ['Yes', 'No'],
                                   help="Do you have a history of skin cancer?")
    with col2:
        Other_Cancer = st.selectbox('Other Cancer', ['Yes', 'No'],
                                    help="Do you have a history of other types of cancer?")
    with col3:        
        Depression = st.selectbox('Depression', ['Yes', 'No'],
                                  help="Do you suffer from depression?")
    with col1:
        Diabetes = st.selectbox('Diabetes', ['Yes', 'No'],
                                help="Do you have diabetes?")
    with col2:
        Arthritis = st.selectbox('Arthritis', ['Yes', 'No'],
                                 help="Do you have arthritis?")
    with col3:
        Sex = st.selectbox('Gender', ['Male', 'Female'],
                           help="Select your gender.")
    with col1:
        Age_Category = st.selectbox('Age Category', ['70-74', '60-64', '75-79', '80+', '65-69', '50-54', '45-49',
                                                     '18-24', '30-34', '55-59', '35-39', '40-44', '25-29'],
                                    help="Select your age category.")
    with col2:
        Height_cm = st.number_input('Height (cm)', help="Enter your height in centimeters.")
    with col3:
        Weight_kg = st.number_input('Weight (kg)', help="Enter your weight in kilograms.")
    with col1:
        BMI = st.number_input('BMI', help="Calculate your BMI (Body Mass Index).")
    with col2:
        Smoking_History = st.selectbox('Smoking', ['Yes', 'No'],
                                       help="Do you smoke or have a history of smoking?")
    with col3:
        Alcohol_Consumption = st.number_input('Alcohol Consumption value',
                                              help="Enter your alcohol consumption value.")
    with col1:
        Fruit_Consumption = st.number_input('Fruit Consumption value',
                                            help="Enter your daily fruit consumption value.")
    with col2:
        Green_Vegetables_Consumption = st.number_input('Green Vegetables Consumption value',
                                                       help="Enter your daily green vegetables consumption value.")
    with col3:
        FriedPotato_Consumption = st.number_input('Fried Potato Consumption value',
                                                  help="Enter your consumption value of fried potatoes.")
    
    if General_Health == 'Excellent':
        General_Health = 0
    elif General_Health == 'Fair':
        General_Health = 1
    elif General_Health == 'Good':
        General_Health = 2
    elif General_Health == 'Poor':
        General_Health = 3
    else:
        General_Health = 4
    
    if Checkup == '5 or more years ago':
        Checkup = 0
    elif Checkup == 'Never':
        Checkup = 1
    elif Checkup == 'Within the past 2 years':
        Checkup = 2
    elif Checkup == 'Within the past 5 years':
        Checkup = 3
    else:
        Checkup = 4
    
    Exercise = 1 if Exercise == 'Yes' else 0
    Skin_Cancer = 1 if Skin_Cancer == 'Yes' else 0
    Other_Cancer = 1 if Other_Cancer == 'Yes' else 0
    Depression = 1 if Depression == 'Yes' else 0
    Diabetes = 1 if Diabetes == 'Yes' else 0
    Arthritis = 1 if Arthritis == 'Yes' else 0
    Sex = 1 if Sex == 'male' else 0
    Smoking_History = 1 if Smoking_History == 'Yes' else 0
    
    age_categories = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+']
    Age_Category = age_categories.index(Age_Category)
    
    if st.button('Predict Heart Disease'):
        
        input_data = ([General_Health, Checkup, Exercise, Skin_Cancer,                       
                      Other_Cancer, Depression, Diabetes, Arthritis, Sex,
                      Age_Category, Height_cm, Weight_kg, BMI, Smoking_History,                   
                      Alcohol_Consumption, Fruit_Consumption, Green_Vegetables_Consumption, FriedPotato_Consumption])
    
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        heart_disease_prediction = Heart_Disease_model.predict(input_data_reshaped)
        
        if heart_disease_prediction == [1]:
          st.success(f"Cardiovascular disease Detected")
        else:
          st.success(f"No Cardiovascular disease detected")
            
if selected == 'Parkinson':
    st.title('Parkinson Prediction')
    # Input fields
    # columns for Input fields 
    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_Fo = st.number_input('MDVP Fo(Hz)', help = "Enter the MDVP value in Fo(Hz)")        
    with col2:    
        MDVP_Fhi = st.number_input('MDVP Fhi(Hz)',help = "Enter the MDVP value in Fhi(Hz)")       
    with col3:    
        MDVP_Flo = st.number_input('MDVP Flo(Hz)',help = "Enter the MDVP value in Flo(Hz)")       
    with col1:    
        MDVP_Jitter = st.number_input('MDVP Jitter(%)',help = "Enter the MDVP value in Jitter(%)")      
    with col2:    
        MDVP_Jitter = st.number_input('MDVP Jitter(Abs)',help = "Enter the MDVP value in Jitter(Abs)")   
    with col3:    
        MDVP_RAP = st.number_input('MDVP RAP',help = "Enter the MDVP value in MDVP Rap(Hz)")            
    with col1:   
        MDVP_PPQ = st.number_input('MDVP PPQ',help = "Enter the MDVP value in MDVP PRQ")           
    with col2:    
        Jitter_DDP = st.number_input('Jitter DDP',help = "Enter the value of Jitter DDP")         
    with col3:    
        MDVP_Shimmer = st.number_input('MDVP Shimmer',help = "Enter the MDVP value in MDVP Shimmer")       
    with col1:   
        MDVP_Shimmer = st.number_input('MDVP Shimmer(dB)',help = "Enter the MDVP value in MDVP Shimmer(dB)")   
    with col2:    
        Shimmer_APQ3 = st.number_input('Shimmer APQ3',help = "Enter the value of Shimmer APQ3")      
    with col3:    
        Shimmer_APQ5 = st.number_input('Shimmer APQ5',help = "Enter the value of Shimmer APQ5")
    with col1:    
        MDVP_APQ = st.number_input('MDVP APQ',help = "Enter the MDVP value in MDVP APQ")      
    with col2:   
        Shimmer_DDA = st.number_input('Shimmer DDA',help = "Enter the value of Shimmer DDA")        
    with col3:    
        NHR = st.number_input('NHR',help = "Enter the value of NHR")                
    with col1:    
        HNR = st.number_input('HNR',help = "Enter the value of HNR")                                
    with col2:    
        RPDE = st.number_input('RPDE',help = "Enter the value of RPDE")            
    with col3:   
        DFA = st.number_input('DFA',help = "Enter the value of DFA")                
    with col1:    
        spread1 = st.number_input('bmi',help = "Enter the value of bmi")          
    with col2:    
        spread2 = st.number_input('spread2',help = "Enter the value of spread2")            
    with col3:    
        D2 = st.number_input('D2',help = "Enter the value of D2")                
    with col1:    
        PPE = st.number_input('PPE',help = "Enter the value of PPE") 
        
    if st.button('Parkinson Result'):
        input_data =([ MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter,
         MDVP_Jitter, MDVP_RAP, MDVP_PPQ, Jitter_DDP,
         MDVP_Shimmer, MDVP_Shimmer, Shimmer_APQ3, Shimmer_APQ5,
         MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA,
         spread1, spread2, D2, PPE])
        
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        prediction = parkinson_model.predict(input_data_reshaped)

        if len(prediction) == 1:
            prediction = 'Enter the values'  # Convert to single value if it's a single-element array
            st.success(f'{prediction}')
            
        if prediction == [1]:
          st.success("parkinson Detected")
        else:
          st.success("parkinson not detected")
        

if selected == 'Covid-19':
    st.title('Covid-19 Prediction')

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female'], help = "Enter your gender")   
    with col2:
        age_60_and_above = st.selectbox('Age(above 60)',['Yes','No'], help = "Are you older than 60 years")
    with col1:
        caugh = st.number_input('Caugh', min_value=0, max_value=10, value=0,help = "Rate your caugh between 0 to 10")            
    with col2:
        fever = st.number_input('Fever',min_value=0, max_value=10, value=0, help = "Rate your fever between 0 to 10")
    with col1:
        sore_throat = st.number_input('Sore Throat',min_value=0, max_value=10, value=0,  help = "Rate your sore throat between 0 to 10")
    with col2:
        shortness_of_breath = st.number_input('Shortness Of Breath',min_value=0, max_value=10, value=0, help = "Rate your shortness of breath between 0 to 10")
    with col1:
        head_ache = st.number_input('Headache', min_value=0, max_value=10, value=0, help = "Rate your headache between 0 to 10") 
    with col2:
        test_indication = st.selectbox('Test Indication',['Other','Abroad','Contact with confirmed'], help = "Select the type of test indication you are with")
        
    gender = 0 if gender == 'Female' else 1
    age_60_and_above = 0 if age_60_and_above == 'No' else 1
    
    if test_indication == 'Abroad':
        test_indication = 0
    elif test_indication == 'Contact with confirmed':
        test_indication = 1 
    else:
        test_indication = 2   
        
    if st.button('Predict Covid'):
             
        input_data = ([gender, age_60_and_above, caugh, fever, sore_throat, shortness_of_breath, head_ache, test_indication])
         
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        covid_19_prediction = Covid_19.predict(input_data_reshaped)
             
        if covid_19_prediction == [1]:
            st.success(f"Covid positive")
        else:
            st.success(f"Covid negitive")
              
if selected == 'Migraine':
    st.title('Migraine and its type detection')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input('Age', min_value=0, max_value=120, help="Enter your age in years.")     
    with col2:
        Duration = st.number_input('Duration', min_value=0, max_value=100, value=0, help="Duration of migraine in hours.") 
    with col3:
        Frequency = st.number_input('Frequency', min_value=0, max_value=100, value=0, help="Frequency of migraines per month.")    
    with col1:
        Location = st.number_input('Location', min_value=0, max_value=10, value=0, help="Pain location score (0-10).")       
    with col2:
        Character = st.number_input('Character', min_value=0, max_value=10, value=0, help="Characteristic of pain (0-10).")       
    with col3:
        Intensity = st.number_input('Intensity', min_value=0, max_value=10, value=0, help="Intensity of pain (0-10).") 
    with col1:
        Nausea = st.number_input('Nausea', min_value=0, max_value=10, value=0, help="Severity of nausea (0-10).")       
    with col2:
        Vomit = st.number_input('Vomit', min_value=0, max_value=10, value=0, help="Frequency of vomiting (0-10).")
    with col3:
        Phonophobia = st.number_input('Phonophobia', min_value=0, max_value=10, value=0, help="Sensitivity to sound (0-10).")
    with col1:
        Photophobia = st.number_input('Photophobia', min_value=0, max_value=10, value=0, help="Sensitivity to light (0-10).")     
    with col2:
        Visual = st.number_input('Visual', min_value=0, max_value=10, value=0, help="Visual disturbances (0-10).")     
    with col3:
        Sensory = st.number_input('Sensory', min_value=0, max_value=10, value=0, help="Sensory disturbances (0-10).") 
    with col1:
        Dysphasia = st.number_input('Dysphasia', min_value=0, max_value=10, value=0, help="Difficulty speaking (0-10).")     
    with col2:
        Dysarthria = st.number_input('Dysarthria', min_value=0, max_value=10, value=0, help="Difficulty articulating words (0-10).")
    with col3:
        Vertigo = st.number_input('Vertigo', min_value=0, max_value=10, value=0, help="Sensation of spinning (0-10).")
    with col1:
        Tinnitus = st.number_input('Tinnitus', min_value=0, max_value=10, value=0, help="Ringing in ears (0-10).")     
    with col2:
        Hypoacusis = st.number_input('Hypoacusis', min_value=0, max_value=10, value=0, help="Reduced hearing (0-10).")
    with col3:
        Diplopia = st.number_input('Diplopia', min_value=0, max_value=10, value=0, help="Double vision (0-10).")
    with col1:
        Defect = st.number_input('Defect', min_value=0, max_value=10, value=0, help="Any defect (0-10).")     
    with col2:
        Ataxia = st.number_input('Ataxia', min_value=0, max_value=10, value=0, help="Lack of muscle coordination (0-10).")
    with col3:
        Conscience = st.number_input('Conscience', min_value=0, max_value=10, value=0, help="Level of consciousness (0-10).")
    with col1:
        Paresthesia = st.number_input('Paresthesia', min_value=0, max_value=10, value=0, help="Tingling or numbness (0-10).")
    with col2:
        DPF = st.number_input('DPF', min_value=0, max_value=10, value=0, help="Days per month with functional impairment (0-10).")
        
    if st.button('Detect Migraine'):
        input_data = [Age, Duration, Frequency, Location, Character, Intensity, Nausea, Vomit,
                      Phonophobia, Photophobia, Visual, Sensory, Dysphasia, Dysarthria, Vertigo,
                      Tinnitus, Hypoacusis, Diplopia, Defect, Ataxia, Conscience, Paresthesia, DPF]
        
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1, -1)
        
        # Ensure Migraine_model is loaded correctly before calling predict
        migraine_prediction = Migraine_model.predict(input_data_reshaped)
        
        if len(migraine_prediction) == 1:
            migraine_prediction = 'No Migraine'  # Convert to single value if it's a single-element array
            st.success(f'{migraine_prediction}')
            
        else:    
            if migraine_prediction == 0:
                st.success("Basilar-type aura")
            elif migraine_prediction == 1:
                st.success("Familial hemiplegic migraine")
            elif migraine_prediction == 2:
                st.success("Migraine without aura")
            elif migraine_prediction == 3:
                st.success("Other")
            elif migraine_prediction == 4:
                st.success("Sporadic hemiplegic migraine")
            elif migraine_prediction == 5:
                st.success("Typical aura with migraine")
            else:
                st.success("Typical aura without migraine")
                
if selected == 'Alzheimer':
    st.title('Alzheimer Detection')
    #input fields
    #columns for Input fields
    col1,col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age',min_value=0, max_value=120, value=0, help = "Enter your Age")
    with col2:
        MF = st.selectbox('Gender',['Male','Female'], help = "Select your gender")
    with col1:
        ASF = st.number_input('ASF', value=0.000, help = "Enter your ASF value")
    with col2:
        SES = st.number_input('SES', value=0.0, help = "Enter your SES value")
    with col1:
        MMSE = st.number_input('MMSE', value=0.0, help = "Enter your MMSE value")
    with col2:
        CDR = st.number_input('CDR', value=0.0, help = "Enter your CDR value")
    with col1:
        eTIV = st.number_input('eTIV', min_value=0, max_value=5000, value=0, help = "Enter your eTIV value")
    with col2:
        nWBV = st.number_input('nWBV', help = "Enter your nWBV value")
         
    MF = 0 if MF == 'Female' else 1
    
    #Button for prediction
    if st.button('Alzheimer Result'):
        #Create input array
        input_data = ([age, MF, ASF, SES, MMSE, CDR, eTIV, nWBV])
        
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        prediction = Alzheimer_model.predict(input_data_reshaped)

        if prediction == [2]:
            st.success(f"Non Demented: No Alzheimer's Disease" )
        elif prediction == [1]:
            st.success(f"Demented: The patient has Alzheimer's Disease")
        else:
            st.success(f"Converted: Consist Symptoms of Alzhimer's Disease, not fully effected!")
    
if selected == 'Depression & Anxiety':
    #page title
    st.title('Depression & Anxiety Detection')
    
    # Input fields
    # columns for Input fields 
    col1, col2 = st.columns(2)
    with col1: 
        age = st.number_input('Age', min_value=0, max_value=120, value=0, help = "Enter your Age")
    with col2:             
        gender = st.selectbox('Gender',['male','female'], help = "Enter you Gender")   
    with col1:               
        bmi = st.number_input('bmi', help = "Enter your bmi value")
    with col2:                   
        who_bmi = st.selectbox('who bmi',['Class I Obesity', 'Normal', 'Overweight', 'Not Availble',
                    'Class III Obesity', 'Underweight', 'Class II Obesity'], help = "Select your type of WHO bmi")    
    with col1:               
        phq_score = st.number_input('phq score', help = 'Enter you phq score')
    with col2:              
        depression_severity = st.selectbox('depression severity',['Mild', 'Moderately severe', 'None-minimal', 'Moderate','Severe', 'none'], help = "Select your depression severity")     
    with col1:
        depressiveness = st.selectbox('depressiveness',['Yes', 'No'], help = "Are you feeling depressiveness")          
    with col2:
        suicidal = st.selectbox('suicidal',['Yes', 'No'], help = "are you feeling to attempt suicidal act")                 
    with col1:
        depression_treatment = st.selectbox('depressiveness treatment',['Yes', 'No'], help =  "Have you ever had Depression treatment")    
    with col2:
        gad_score = st.number_input('gad score', help = "what is gad score")              
    with col1:
        anxiety_severity = st.selectbox('anxiety severity',['Moderate', 'Mild', 'Severe', 'None-minimal', '0'], help = "Select your anxiety severity")        
    with col2:
        anxiousness = st.selectbox('anxiousness',['Yes', 'No'],help = "Are you feeling anxiousness")                    
    with col1:
        anxiety_treatment = st.selectbox('anxiety treatment',['Yes', 'No'], help = "Have you ever had anxiety treatment")      
    with col2:
        epworth_score = st.number_input('epworth score', help = "what is your epworth score")         
    with col1:
        sleepiness = st.selectbox('sleepiness',['Yes', 'No'],help = "can you able to sleep well") 
        
    gender = 0 if gender == 'female' else 1
    depressiveness = 0 if depressiveness == 'No' else 1
    suicidal = 0 if suicidal == '' else 1
    depression_treatment = 0 if depression_treatment == 'No' else 1
    anxiousness = 0 if anxiousness == 'No' else 1
    anxiety_treatment = 0 if anxiety_treatment == 'No' else 1
    sleepiness = 0 if sleepiness == 'No' else 1
    
    if who_bmi == 'Class I Obesity': 
        who_bmi = 0
    elif who_bmi == 'Class II Obesity':
        who_bmi = 1
    elif who_bmi == 'Class III Obesity':
        who_bmi = 2
    elif who_bmi == 'Normal':
        who_bmi = 3   
    elif who_bmi == 'Not Availble':
        who_bmi = 4
    elif who_bmi == 'Overweight':
        who_bmi = 5
    else:
        who_bmi = 6
        
    if depression_severity == 'Mild': 
        depression_severity = 0
    elif depression_severity == 'Moderate':
        depression_severity = 1
    elif depression_severity == 'Moderately severe':
        depression_severity = 2
    elif depression_severity == 'None-minimal':
        depression_severity = 3
    elif depression_severity == 'Severe':
            depression_severity = 4
    else:
        depression_severity = 5  
        
    if anxiety_severity == '0': 
        anxiety_severity = 0
    elif anxiety_severity == 'Mild':
        anxiety_severity = 1
    elif anxiety_severity == 'Moderate':
        anxiety_severity = 2
    elif anxiety_severity == 'None-minimal':
        anxiety_severity = 3   
    else:
        anxiety_severity = 4    
   
# Button for prediction

    if st.button('Depression and Anxiety Result'):
        input_data =([ age, gender,bmi, who_bmi, phq_score,
              depression_severity, depressiveness, suicidal,depression_treatment, gad_score,
               anxiety_severity, anxiousness,anxiety_treatment, epworth_score, sleepiness])
        
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1,-1)
        prediction = depression_anxiety_data.predict(input_data_reshaped)
        if prediction==['1 1']:
          st.success(f"Both Depression and Anxiety Predicted")
        elif prediction==['1 0']:
          st.success(f"Depression Predicted")
        elif prediction ==['0 1']:
          st.success(f"Anxiety Predicted")
        else:
          st.success(f"No Depression or Anxiety Predicted")  
 
if selected == 'Kidney Disease':
   #page title
   st.title('Kidney disease detection')
   col1, col2, col3 = st.columns(3)
   with col1:
       age = st.number_input('Age', min_value=0, max_value=120, value=0,help = "Enter your age")               
   with col2:    
    bp = st.number_input('Blood Pressure (bp)', help="Enter your blood pressure level")                
    with col3:    
        sg = st.number_input('Specific Gravity (sg)', help="Enter your specific gravity level")                
    with col1:    
        al = st.number_input('Albumin (al)', help="Enter your albumin level")                
    with col2:    
        su = st.number_input('Sugar (su)', help="Enter your sugar level")                
    with col3:    
        rbc = st.selectbox('Red Blood Cells (rbc)', ['normal', 'abnormal'], help="Select if red blood cells are normal or abnormal")                
    with col1:    
        pc = st.selectbox('Pus Cells (pc)', ['normal', 'abnormal'], help="Select if pus cells are normal or abnormal")                 
    with col2:    
        bu = st.number_input('Blood Urea (bu)', help="Enter your blood urea level")                
    with col3:    
        sc = st.number_input('Serum Creatinine (sc)', help="Enter your serum creatinine level")                
    with col1:    
        sod = st.number_input('Sodium (sod)', help="Enter your sodium level")               
    with col2:    
        pot = st.number_input('Potassium (pot)', help="Enter your potassium level")               
    with col3:    
        hemo = st.number_input('Hemoglobin (hemo)', help="Enter your hemoglobin level")              
    with col1:    
        pcv = st.number_input('Packed Cell Volume (pcv)', help="Enter your packed cell volume")                 
    with col2:    
        wc = st.number_input('White Blood Cell Count (wc)', help="Enter your white blood cell count")                 
    with col3:    
        htn = st.selectbox('Hypertension (htn)', ['Yes', 'No'], help="Select if you have hypertension")                
    with col1:    
        dm = st.selectbox('Diabetes Mellitus (dm)', ['Yes', 'No'], help="Select if you have diabetes mellitus")                 
    with col2:    
        cad = st.selectbox('Coronary Artery Disease (cad)', ['Yes', 'No'], help="Select if you have coronary artery disease")                
    with col3:    
        pe = st.selectbox('Pedal Edema (pe)', ['Yes', 'No'], help="Select if you have pedal edema")                 
    with col1:    
        ane = st.selectbox('Anemia (ane)', ['Yes', 'No'], help="Select if you have anemia")
  
   rbc = 0 if rbc == 'abnormal' else 1
   pc = 0 if pc == 'abnormal' else 1
   htn = 0 if htn == 'No' else 1
   dm = 0 if dm == 'No' else 1
   cad = 0 if cad == 'No' else 1
   pe = 0 if pe == 'No' else 1
   ane = 0 if ane == 'No' else 1
   
   if st.button('Kidney disease Result'):
       input_data = ([age, bp, sg, al, su,rbc, pc, bu, sc, sod, pot,
                        hemo, pcv, wc, htn, dm, cad, pe, ane])
       input_data_as_np_array = np.asarray(input_data)
       input_data_reshaped = input_data_as_np_array.reshape(1,-1)
       prediction = kidney_disease_model.predict(input_data_reshaped)

       if prediction == [1]:
         st.success("kidney disease Detected")
       else:
         st.success("kidney disease not detected")
           