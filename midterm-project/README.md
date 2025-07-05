# Midterm Project - Heart Disease Diagnosis Predictor
![image](https://github.com/user-attachments/assets/1917ed60-ca8c-47be-99cb-5cbef03adf77)


## ❓ Problem Statement
Heart disease is the biggest killer in the UK - that's something that really struck me when I started researching this project. We're looking at tens of thousands of deaths each year, and the frustrating thing is that many of these could probably be prevented if we spotted the warning signs earlier.

The trouble is, working out who's going to develop heart problems isn't exactly straightforward. Of course, doctors have ways to test for it, but these tests are costly and take ages. Plus, most people don't even consider getting checked until they're already feeling unwell. By that point, you're often dealing with serious damage that could have been avoided altogether.

I've been thinking about this problem from a practical angle. What if we could use the information GPs already collect during routine appointments to better predict who's at risk? Rather than waiting for someone to turn up at A&E with chest pains, we could identify potential problems much sooner. This would give doctors and patients a proper chance to take action before things get serious.

This is where machine learning comes in handy. I decided to have a go at this challenge using a dataset from the Cleveland Clinic that contains medical records from 303 patients. The data includes things like age, blood pressure, cholesterol levels, and other measurements that doctors typically record. My aim is to build a model that can look at these factors and accurately predict whether someone is likely to have heart disease.

## 🎯 Goals

The main goal is to build an end-to-end Machine Learning project for the Mid-Term Project for the ML Zoomcamp. 
We have been instructed to:
- choose interesting dataset
- load data, conduct exploratory data analysis (EDA), clean it
- train & test ML model(s)
- deploy the model (as a web service) using containerization

## ℹ️ Dataset Characteristics

- Number of Instances: 303 patients
- Number of Attributes: 14 (out of an original 76 attributes, 14 are typically used in published experiments)
- Associated Task: Classification (predicting the presence or absence of heart disease)
- Attribute Types: Categorical, Integer, Real
- Missing Values: Original Dataset contains some missing value. Dataset I used from Kaggle had Missing Values replaced.




**Command Line Code to Use Production Level Server When Running Predict App via Flask**
```
gunicorn --bind 0.0.0.0:9696 predict:app
```
**Command Line Code to Use Production Level Server When Running Predict App in PipEnv**

```
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```

