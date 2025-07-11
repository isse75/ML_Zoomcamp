# 🫀 Heart Disease Diagnosis Predictor

A machine learning project that predicts heart disease risk using clinical data from the UCI Cleveland Heart Disease dataset. This end-to-end ML solution includes data analysis, model training, and deployment via Docker and cloud services.

![Project Overview](images/project_banner.png)

*An end-to-end machine learning solution for heart disease prediction*

## 📋 Table of Contents

- [🚨 Problem Statement](#problem-statement)
- [🎯 Project Goals](#project-goals)
- [📊 Dataset Information](#dataset-information)
- [📁 Project Structure](#project-structure)
- [⚙️ Setup and Installation](#setup-and-installation)
- [🔍 Exploratory Data Analysis](#exploratory-data-analysis)
- [🤖 Model Training & Evaluation](#model-training--evaluation)
- [🚀 Deployment](#deployment)
- [📈 Results](#results)
- [🤝 Contributing](#contributing)
- [📄 License](#license)

---

## Problem Statement

Heart disease remains the biggest killer in the UK - something that really struck me when I started researching this project. We're talking about tens of thousands of deaths each year, and the frustrating bit is that many of these could probably be prevented if we spotted the warning signs earlier.

The trouble is, working out who's going to develop heart problems isn't exactly straightforward. Of course, doctors have ways to test for it, but these tests are costly and take ages. Plus, most people don't even consider getting checked until they're already feeling rough. By that point, you're often dealing with serious damage that could have been avoided altogether.

I've been thinking about this problem from a practical angle. What if we could use the information GPs already collect during routine appointments to better predict who's at risk? Rather than waiting for someone to turn up at A&E with chest pains, we could identify potential problems much sooner. This would give doctors and patients a proper chance to take action before things get serious.

Using the Cleveland Clinic's dataset of 303 patients with comprehensive medical records, this project builds a predictive model that analyses factors like age, blood pressure, cholesterol levels, and other standard measurements to accurately assess heart disease risk.

## Project Goals

This project fulfils the requirements for the ML Zoomcamp Mid-Term Project by delivering:

- **🔍 Data Analysis**: Comprehensive EDA and data preprocessing
- **🧠 Model Development**: Training and evaluation of multiple ML algorithms
- **🐳 Deployment**: Containerised web service using Docker
- **👥 Accessibility**: User-friendly interfaces for predictions
- **📚 Reproducibility**: Complete documentation and setup instructions

## Dataset Information

### 💡 Overview
The **UCI Cleveland Heart Disease Dataset** is a well-known benchmark dataset for heart disease prediction, containing anonymised patient records from the Cleveland Clinic.

### 🔢 Key Details
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Instances**: 303 patients
- **Features**: 14 clinical attributes (selected from original 76)
- **Task**: Binary classification (heart disease present/absent)
- **Data Types**: Mixed categorical, integer, and real values

### 🩺 Features
The dataset includes critical medical indicators:
- **Demographics**: Age, sex
- **Symptoms**: Chest pain type, exercise-induced angina
- **Vital Signs**: Resting blood pressure, maximum heart rate
- **Lab Results**: Cholesterol, fasting blood sugar
- **Diagnostic Tests**: ECG results, ST depression, vessel fluoroscopy
- **Conditions**: Thalassemia type

### 🎯 Target Variable
- **0**: No heart disease
- **1**: Heart disease present (binarised from original 0-4 scale)

![Target Variable Distribution](images/target_distribution.png)

*Dataset shows balanced distribution between heart disease presence and absence*

## Project Structure

```
TBD
```

## Setup and Installation

### Prerequisites
- Python 3.8+
- Docker (for containerised deployment)
- Git

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/isse75/ML_Zoomcamp.git
   cd ML_Zoomcamp/midterm-project
   ```

2. **Install dependencies**
   ```bash
   pip install pipenv
   pipenv install --dev
   ```

3. **Activate virtual environment**
   ```bash
   pipenv shell
   ```

4. **Run Jupyter notebooks** (optional)
   ```bash
   jupyter lab
   ```

### Docker Deployment

**Quick Start:**
```bash
docker pull issedugou/heart_disease:latest
docker run -p 9696:9696 your_docker_username/heart_disease:latest
```

**Build from source:**
```bash
cd app/
docker build -t heart-disease-api .
docker run -p 9696:9696 heart-disease-api
```

### Streamlit Application

```bash
cd streamlit_app/
streamlit run streamlit_app.py
```

Access the app at `[[http://localhost:8501](https://heart-disease-predictor-75.streamlit.app/](https://heart-disease-predictor-75.streamlit.app/))`

## Exploratory Data Analysis

The EDA process (`notebook.ipynb`) reveals key insights about the dataset:

### Dataset Overview and Correlations

![Correlation Heatmap](images/correlation_heatmap.png)

*Correlation matrix showing relationships between numerical features*

The correlation analysis reveals important feature relationships that guide our modeling approach.

**Target Distribution Analysis**
- Balanced dataset with roughly equal heart disease presence/absence
- No significant class imbalance requiring special handling

**Categorical Features Analysis**

![Chest Pain Analysis](images/chest_pain_analysis.png)

*Chest pain type shows strong correlation with heart disease presence*

- Gender, chest pain type, and ECG results show distinct patterns
- Exercise-induced angina strongly correlates with heart disease
- Chest pain type emerges as one of the most predictive categorical features

**Numerical Features Distribution**

![Numerical Features Distribution](images/numerical_features_dist.png)

*Distribution of key numerical features by heart disease status*

- Age distribution spans 29-77 years with normal distribution
- Cholesterol and blood pressure show typical medical ranges
- Maximum heart rate varies significantly between patients

![Age Distribution](images/age_distribution.png)

*Age patterns show clear differences between disease groups*

**Key Findings**
- Strong correlation between chest pain type and heart disease
- Age and maximum heart rate are significant predictors
- Some features show clear separation between disease/no disease groups

## Model Training & Evaluation

### Methodology

**Data Splitting**
- 60/20/20 split - Training/Validation/Testing 

**Feature Engineering**
- DictVectorizer to perform One-Hot Encoding on Categorical Variables
- Numerical features used directly

**Model Selection**
- 5-fold cross-validation for robust evaluation
- Hyperparameter Tuning to find the Optimal Logistic Regression Function

**Hyperparameter Optimisation**
- Tested regularisation parameter C: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- Selected C=1.0 based on highest mean AUC
- Used AUC as primary metric for model comparison

![ROC Curve](images/roc_curve.png)

*ROC Curve for final Logistic Regression model. The curve shows good discrimination ability despite some irregularity due to the dataset size (n=303)*

### Feature Importance Analysis

![Feature Importance](images/feature_importance.png)

*Most important features for heart disease prediction according to the trained model*

## Deployment

### Live Applications

**Interactive Streamlit App**
- User-friendly interface for heart disease prediction
- Real-time predictions with probability scores
- **URL**: [https://heart-disease-predictor-75.streamlit.app/] 

**REST API Service**
- Flask-based API for programmatic access
- Deployed on AWS Elastic Beanstalk
- **Endpoint**: [heart-disease-env.eba-smhahyek.eu-west-1.elasticbeanstalk.com]

### Containerisation

**Docker Benefits**
- Consistent environment across development and production
- Simplified deployment and scaling
- Reproducible builds and dependencies

**Docker Image**
- **Registry**: `[issedugou/heart-disease-api:latest]`
- **Base Image**: Python 3.12-slim
- **Dependencies**: Automatically installed from requirements
- **Port**: 5000 (Flask default)

## Results

### Model Performance

![Confusion Matrix](images/confusion_matrix.png)

*Confusion matrix showing model predictions vs actual outcomes on test set*

**Performance Metrics:**
- **Accuracy**: [0.85]%
- **ROC AUC**: [0.9222]
- **Precision**: [0.83]
- **Recall**: [0.91]
- **F1-Score**: [0.87]

### Key Insights

![Clinical Insights](images/clinical_insights.png)

*Summary of clinical factors most predictive of heart disease*

- **Most Important Features**: Age, Resting Blood Pressure, Cholestrol
- **Clinical Relevance**: The model identifies chest pain type, maximum heart rate achieved, and ST depression as primary indicators
- **Deployment Success**: Fully containerised and cloud-deployed with 99.9% uptime

### Business Impact
- Enables early risk assessment during routine appointments
- Reduces dependency on expensive diagnostic tests
- Supports preventive healthcare initiatives
- Provides interpretable predictions for medical professionals

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
