
#import required libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('diabetes.csv')

#headings
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Statistics')
st.write(df.describe())


x = df.drop(['Outcome'], axis = 1)
y = df['Outcome']

#standardization of data
scaler = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
x = standardized_data

#splitting the training and test data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 2)


#function to take input from slider and convert it into dataframe
def patientReport():
  pregnancies = st.sidebar.slider('Pregnancies', 0,18, 5)
  glucose = st.sidebar.slider('Glucose', 0,200, 100)
  bp = st.sidebar.slider('Blood Pressure', 0,125, 65)
  skinthickness = st.sidebar.slider('Skin Thickness', 0,100, 23)
  insulin = st.sidebar.slider('Insulin', 0,850, 74)
  bmi = st.sidebar.slider('BMI', 0,70, 25)
  dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0,3.0, 0.41)
  age = st.sidebar.slider('Age', 21,90, 30)

  patient_report_data = {
      'Pregnancies':pregnancies,
      'Glucose':glucose,
      'BloodPressure':bp,
      'SkinThickness':skinthickness,
      'Insulin':insulin,
      'BMI':bmi,
      'DiabetesPedigreeFunction':dpf,
      'Age':age
  }
  report_data = pd.DataFrame(patient_report_data, index=[0])
  return report_data


#displaying the patient data
user_data = patientReport()
st.subheader('Patient Report Data')
st.write(user_data)


#Train the model using SVM
svm_classifier = svm.SVC(kernel='linear')
svm_classifier.fit(x_train, y_train)

#standardize the user data
user_data_standardization = scaler.transform(user_data)
#make prediction on user data
user_result = svm_classifier.predict(user_data_standardization)


#visualization
st.title('Visualised Patient Report')

#initializing  the color of the user a/c to their current diabetic condition
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

#Age vs Pregnancies
st.header('Pregnancy count Graph (Others vs Yours)')
Pregnancy_fig = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'pastel')
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Pregnancies'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.legend(["Healthy","Unhealthy"])
st.pyplot(Pregnancy_fig)



#Age vs Glucose
st.header('Glucose Value Graph (Others vs Yours)')
glucose_fig = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='Blues')
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.legend(["Healthy","Unhealthy"])
st.pyplot(glucose_fig)


#Age vs Bp
st.header('Blood Pressure Value Graph (Others vs Yours)')
bp_fig = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['BloodPressure'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.legend(["Healthy","Unhealthy"])
st.pyplot(bp_fig)


#Age vs St
st.header('Skin Thickness Value Graph (Others vs Yours)')
skin_thickness_fig = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='YlOrBr')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['SkinThickness'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.legend(["Healthy","Unhealthy"])
st.pyplot(skin_thickness_fig)


#Age vs Insulin
st.header('Insulin Value Graph (Others vs Yours)')
insulin_fig = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='spring')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Insulin'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.legend(["Healthy","Unhealthy"])
st.pyplot(insulin_fig)


#Age vs BMI
st.header('BMI Value Graph (Others vs Yours)')
bmi_fig = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='magma')
ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['BMI'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.legend(["Healthy","Unhealthy"])
st.pyplot(bmi_fig)


#Age vs Dpf
st.header('DPF Value Graph (Others vs Yours)')
dpf_fig = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='rainbow')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['DiabetesPedigreeFunction'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.legend(["Healthy","Unhealthy"])
st.pyplot(dpf_fig)


#Final prediction
st.subheader('Your Report: ')
result=''
if user_result[0]==0:
  result = 'You are not Diabetic'
else:
  result = 'You are Diabetic'
st.title(result)
st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, svm_classifier.predict(x_test))*100)+'%')


