# building the model

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # load dataset (iris)
# def load_data():
#     iris = sns.load_dataset('iris')
#     return iris

# iris_data = load_data()

# # write something on streamlit
# st.write(iris_data)
# # contrl s to save
# # change directory at terminal - cd streamlit_sample
# # run it; streamlit run iris_app.py : when you see email, click enter

# # creating sidebar for feature selection
# st.sidebar.subheader("Select features for Bivariate Plot")
# #select all features except the last one (the target)
# x_feature = st.sidebar.selectbox('X-axis Feature', 
#                                  iris_data.columns[:-1])
# # do same for y
# y_feature = st.sidebar.selectbox('Y-axis Feature', 
#                                  iris_data.columns[:-1])

# target_cat = st.sidebar.selectbox('Target Categories',
#                                    iris_data['species'].unique())

# st.subheader('Segmenting data')
# segment_data = iris_data[iris_data['species']== target_cat]
# st.write(segment_data)


# # Create a bivariate scatter plot
# st.write(f"### Bivariate Plot between {x_feature} and {y_feature}")
# fig, ax = plt.subplots()
# sns.scatterplot(data = iris_data, x = x_feature, y = y_feature, 
#                 hue = "species")
# plt.xlabel(x_feature)
# plt.ylabel(y_feature)
# st.pyplot(fig)
# #  Filtering the dataset based on the user input
# ## Create a slider for selecting the sepal length range
# st.subheader("Creating Slider")
# sepal_length_min= st.slider("Minimum Sepal Length", min_value= 0.0,
#                              max_value=10.0,value = 5.2)

# sepal_length_max= st.slider("Maximum Sepal Length", min_value= 0.0,
#                              max_value=10.0,value = 10.0)
# st.write(sepal_length_min) # to display whatever vale the user picks
# st.write(sepal_length_max)

# # select species using a radio button
# species = st.radio('select a specie', iris_data['species'].unique())

# filtered_data = iris_data[
#     (iris_data["sepal_length"] >= sepal_length_min) &
#     (iris_data["sepal_length"] <= sepal_length_max)| #or
#      (iris_data["species"] == species)]
# #Filter the dataset and display to the screen
# st.write(filtered_data)


# ##File Uploads in Streamlit
# st.subheader("File Upload and Display")
# uploaded_file = st.file_uploader("Upload a File", type = ["pdf", 
#                                                           "png", "jpg", "jpeg"])
# # check if the file is properly uploaded
# if uploaded_file is not None:
#     st.write("Filename: ", uploaded_file.name) 
#     st.write("File type: ", uploaded_file.type)
#     st.image (uploaded_file)


# # Streamlit Forms and Button
# st.subheader("Streamlit Form and Button")
# # using context manager for form - with
# with st.form("User Input Form"):
#     #Add the text input
#     name = st.text_input("Enter your full name")
#     email = st.text_input("Enter your email")
#     department = st.selectbox("Select your department", ["HR", "Finance", "Engineering", "Marketing"])
#     age = st.slider("Select your age", min_value=18, max_value=90, value=25)
#     submit_button = st.form_submit_button("Submit")

# # if the user clicks the button
# if submit_button: 
#     #display user input
#     st.write("Name:", name)
#     st.write("Email:", email)
#     st.write("Department:", department)
#     st.write("Age:", age)


# # case study : training a model using pycaret
# # Data = inbuilt pycaret income data
# import pandas as pd
# from pycaret.datasets import get_data
# from pycaret.classification import setup, compare_models, finalize_model
# from pycaret.classification import automl, save_model

# #get the data
# df = get_data("income", save_copy = True, verbose=False)

# def clean_marital_status(status):
#     married_list= ["Married-civ-spouse", "Married-AF-spouse"]
#     if status in married_list:
#         return "Married"
#     else:
#         return "Not married"

# df["marital-status"] = df["marital-status"].apply(lambda x: clean_marital_status(x))
# #rename the target column
# df.rename(columns = {'income >50K': 'income'}, inplace = True)
# df.columns
# # Prepare the data
# s = setup(data = df, target = "income", fix_imbalance=True, 
#           remove_outliers =True, verbose=False)

# #save the transformed data
# transformed_df = s.dataset_transformed
# transformed_df.to_csv("transformed_income.csv", index=False)

# #train different models on the data
# top_models = compare_models(n_select=3, sort = "F1", verbose=False)

# # select the best model
# best_model = automl(optimize = 'F1')

# # save the best model
# save_model(best_model, 'best_model', verbose=False)

# building the streamlit app
import os
import pandas as pd
from pycaret.classification import predict_model, load_model

model_dir = "C:/Users/User/streamlit_sample/best_model"
# note that not only the model was saved with pycaret, the pipeline is also saved
# if model was built manually, one hot encoding will make it have too many columns
# if your work involve traditional machine learning and you wish to deploy, always use pycaret
# this will prevent manual definition of the one-hot encoded values; check the transformed csv for more info
model = load_model(model_dir)

#create the streamlit app
st.title("Income Prediction App")

# create a form for user input
st.sidebar.header('User Input Features')

# load the original dataset usef for training
df = pd.read_csv(r'C:\Users\User\streamlit_sample\income.csv')

# since the minor cleaning done was not save in a new csv, let's copy the code again

def clean_marital_status(status):
    married_list= ["Married-civ-spouse", "Married-AF-spouse"]
    if status in married_list:
        return "Married"
    else:
        return "Not married"

df["marital-status"] = df["marital-status"].apply(lambda x: clean_marital_status(x))
#rename the target column
df.rename(columns = {'income >50K': 'income'}, inplace = True)

# define the input fields 
features= df.drop("income", axis = 1) 

#separate the numeric columns from the categorical columns since 
#the widget suitable for each data type are different"""
num_features= features.select_dtypes (include = "number").columns.tolist() 
cat_features= features.select_dtypes (include = "object").columns.tolist() 

# create a dictionary that will be used to create a dataframe out of the input the users supply
input_fields = {}
for feature in num_features: 
    input_fields[feature]= st.sidebar.slider(f"Select {feature}", features[feature].min(), 
                                             features[feature].max(), 0)
    # sidebar is used because we want the slider to be at the side of the app
    # 0 was chosen as the default feature

for feature in cat_features:
    input_fields[feature]= st.sidebar.selectbox(f"Select {feature}", features[feature].unique())    
    # categorical variables dont need a slider so we used box widget
# also there is no maximum or minumum for catergoricals so we go with unique

# Create a dataframe for the user input
user_input = pd.DataFrame([input_fields]) #  creating a dataframne out of what the user selects
income_group = ["<50K", ">50K"]

#Make predictions
if st.sidebar.button("Predict"):
    prediction = predict_model(model, data = user_input, raw_score=True) 
    # raw score is used since it's a classification task. It will display for both classes but 
    # the probalility of the 2 classes
    st.write(prediction)
    predicted_label = prediction ["prediction_label"].iloc[0]
    # prediction_label is a column generated by pycaret
    st.write(f"The Predicted Income Group is: {income_group[predicted_label]}")
    
