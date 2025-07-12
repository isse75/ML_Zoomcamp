# 🏦 Bank Marketing Campaign Classifier

This project predicts whether a bank customer will subscribe to a term deposit, based on direct marketing campaign data. Built with a complete ML workflow from EDA to deployment, it includes interactive visualisation, multiple model comparisons, API services, and a Streamlit UI—deployed with Docker on AWS EC2.

![Project Banner](images/project_banner.png)

*An end-to-end ML pipeline for customer targeting using the UCI Bank Marketing Dataset*

---

## 📋 Table of Contents

* [📌 Problem Statement](#problem-statement)
* [🎯 Project Goals](#project-goals)
* [📊 Dataset Information](#dataset-information)
* [📁 Project Structure](#project-structure)
* [⚙️ Setup and Installation](#setup-and-installation)
* [🔍 Exploratory Data Analysis](#exploratory-data-analysis)
* [🧠 Model Training & Selection](#model-training--selection)
* [🚀 Deployment](#deployment)
* [📈 Results & Feature Importance](#results--feature-importance)
* [🤝 Contributing](#contributing)
* [📄 License](#license)

---

## 📌 Problem Statement

Marketing calls are expensive and time-consuming. However, only a small proportion of customers actually subscribe to financial products like term deposits. This project aims to help banks predict, *before making contact*, which customers are most likely to say yes—so that campaigns can be smarter, leaner, and more effective.

---

## 🎯 Project Goals

* Clean and analyse the UCI Bank Marketing dataset
* Visualise categorical and numerical feature impacts
* Address class imbalance and data leakage issues
* Train and evaluate multiple models (Logistic Regression, Random Forest, XGBoost)
* Select the best model using cross-validation and metric analysis
* Deploy the final model to an API service
* Create an interactive web front end using Streamlit
* Containerise and deploy via Docker to AWS EC2

---

## 📊 Dataset Information

### 🔹 Source

* UCI Machine Learning Repository: [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

### 🔹 Stats

* 41,188 rows
* 20 input features
* Binary target: `y` (yes/no for term deposit subscription)

### 🔹 Feature Groups

* **Demographic**: Age, job, marital, education
* **Financial**: Default, housing loan, personal loan
* **Communication**: Contact type, month, day of week, duration
* **Campaign**: Number of contacts, outcomes of previous attempts

---

## 📁 Project Structure

```
.
├── data                      # Full EDA and model comparison
    └── bank-additional-full.csv        #Bank Marketing Dataset
├── Notebook.ipynb           # Full EDA and model comparison
├── Train.py                 # Training script
├── predict.py               # Inference service
├── streamlit_app.py         # Streamlit web UI
├── requirements.txt         # Dependency Management
├── .dockerignore            # Docker Reference File
├── Dockerfile               # Docker image
├── Pipfile                  # Dependency management
├── Pipfile.lock             # Dependency management
└── XGBoost_model.bin        # Trained model artefact
```

---

## ⚙️ Setup and Installation

### Prerequisites

* Python 3.12+
* Docker
* Pipenv

### Local Setup

```bash
git clone https://github.com/yourusername/bank-campaign-classifier.git
cd bank-campaign-classifier

pip install pipenv
pipenv install
pipenv shell
```

---

## 🔍 Exploratory Data Analysis

Conducted in detail in `Notebook.ipynb`:

* Column cleanup: Replaced `.` with `_` for consistency
* Dropped irrelevant columns: Removed economic indicators (`emp_var_rate`, `cons_price_idx`, etc.)
* Education mapped: 8 values reduced to 4 categories (primary, secondary, tertiary, unknown)
* Target imbalance: Only \~11.3% subscribed → addressed during model evaluation
* Converted `y` to binary: 1 for yes, 0 for no

### Key Visuals and Insights

* `duration` was highly predictive but dropped due to data leakage
* `pdays` removed due to being mostly 999s (uninformative)
* Boxplots and histograms showed trends in `campaign`, `previous`, and other features
* Heatmaps revealed weak linear correlation, justifying use of non-linear models

---

## 🧠 Model Training & Selection

### Models Trained

1. Logistic Regression
2. Random Forest
3. XGBoost

### Techniques Used

* Train/Val/Test split with stratification
* Cross-validation with 5 folds
* Hyperparameter tuning with GridSearchCV
* Threshold selection using F1 optimisation

### Metrics Used

* ROC AUC
* F1 Score
* Precision & Recall
* Accuracy

### Final Model: ✅ XGBoost

* Tuned with `learning_rate`, `max_depth`, `min_child_weight`
* Used `scale_pos_weight` for class imbalance
* Selected based on best F1 and AUC scores on test set

---

## 🚀 Deployment

### Inference API

* FastAPI-style interface hosted on EC2 (port 9696)
* Accepts JSON input and returns binary prediction with probability

### Streamlit Front End

* Built with `streamlit_app.py`
* Interactive form UI
* Submits data to live API
* Displays prediction, probability, and recommendation

📍 **Live App**: [[https://deposit-predictor.streamlit.app/]](https://deposit-predictor.streamlit.app/)

---

## 📈 Results & Feature Importance

### Model Comparison

| Metric    | Logistic Regression | Random Forest | XGBoost  |
| --------- | ------------------- | ------------- | -------- |
| Accuracy  | 0.872               | **0.878**     |   0.873  |
| ROC AUC   | 0.765               |   0.773       | **0.778**|
| F1 Score  | 0.441               |   0.473       | **0.474**|
| Precision | 0.434               | **0.459**     |   0.445  |
| Recall    | 0.447               |   0.488       | **0.508**|

As the XGBoost and Random Forest Models had similar performance, I deceided that XGBoost is the better Model to go ahead with. This is due to the fact that the features in the dataset have relatively low individual correlations with the target variable `y`, as you will be able to see in the next section below. XG Boost Models excel in finding feature interactions and non-linear patterns in datasets. Random Forest Models tend to struggle with weak individual signals. As this dataset also has imbalanced classed, XGBoost's sophistication will be extremely useful.

### Feature Importance (XGBoost)

Top features include:

* `poutcome=success`
* `contact=cellular`
* `month=mar`, `month=jun`, `month=oct`
* `default=no`

These features were most influential in predicting a positive outcome. Marketing timing, communication method, and customer history were key.

---

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

