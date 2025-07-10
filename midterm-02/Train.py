import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, classification_report
from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBClassifier


import warnings
warnings.filterwarnings('ignore')


#data prep
df = pd.read_csv('data/bank-additional-full.csv', delimiter= ';')
df.columns = df.columns.str.replace('.', '_')

df = df.drop(columns=['emp_var_rate', 'cons_price_idx',
       'cons_conf_idx', 'euribor3m', 'nr_employed',
       'pdays', 'duration'])

education_mapping = {
    'illiterate': 'primary',
    'basic.4y': 'primary',
    'basic.6y': 'primary',
    'basic.9y': 'primary',
    'high.school': 'secondary',
    'professional.course': 'secondary',
    'university.degree': 'tertiary',
    'unknown': 'unknown'
}

df['education'] = df['education'].map(education_mapping)

df.y = (df.y=='yes').astype('int')
  
numerical = ['age', 'campaign', 'previous']

categorical = ['job', 'marital', 'education', 'default', 'housing', 
               'loan', 'contact', 'month', 'day_of_week', 'poutcome']


# functions
def split_data(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1, shuffle=True, stratify=df['y'])
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1, shuffle=True, stratify=df_full_train['y'])

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    y_train = df_train.y.values
    y_val = df_val.y.values
    y_test = df_test.y.values
    
    del df_train['y']
    del df_val['y']
    del df_test['y']

    return df_full_train, df_train, df_val, df_test, y_train, y_val, y_test

def transform_data(df): 
    df_full_train, df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
    dicts_train = df_train.to_dict(orient='records')
    dicts_val = df_val.to_dict(orient='records')
    dicts_test = df_test.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    
    X_train = dv.fit_transform(dicts_train)
    X_val = dv.transform(dicts_val)
    X_test = dv.transform(dicts_test)

    return dicts_train, dicts_val, dicts_test, X_train, X_val, X_test


def train(df_train, y_train, model):
    dict_train = df_train.to_dict(orient = 'records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dict_train)
    
    model = model
    model.fit(X_train, y_train)
    
    return dv


def predict(df, dv, model):
    dicts = df.to_dict(orient='records')
    
    X = dv.transform(dicts)
    
    y_pred = model.predict_proba(X)[:,1]
    
    return y_pred

def identify_thresholds(y_val, y_pred):
    thresholds = np.linspace(0,1,101)
    results = []
    
    for t in thresholds:
        y_pred_bin = (y_pred >= t).astype(int)
        precision = precision_score(y_val, y_pred_bin)
        recall = recall_score(y_val, y_pred_bin)
        f1 = f1_score(y_val, y_pred_bin)
        results.append((t, precision, recall, f1))
        
    df_scores = pd.DataFrame(results, columns = ['threshold', 'precision', 'recall', 'f1_score'])
    
    best_row = df_scores.iloc[df_scores['f1_score'].idxmax()]
    
    return best_row
    


#Logistic Regression Training

df_full_train, df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
dicts_train, dicts_val, dicts_test, X_train, X_val, X_test = transform_data(df)


print('Training Logistic Regression')
print()
print('Indentifying Optimal Threshold')
model = LogisticRegression(solver='lbfgs')
dv = train(df_train, y_train, model)
y_pred = predict(df_val, dv, model)
print(identify_thresholds(y_val, y_pred))

print('Training Final Logistic Regression Model')
print()
df_full_train, df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
dicts_train, dicts_val, dicts_test, X_train, X_val, X_test = transform_data(df)
y_full_train = df_full_train['y'].values
df_full_train = df_full_train.drop(columns = ['y'])
dv = train(df_full_train, y_full_train, model)
y_pred = predict(df_test, dv, model)
deposit_decision = (y_pred >= 0.19).astype(int)
auc = roc_auc_score(y_test, y_pred)
print(f'Auc of the final Logistic Regression model = {auc}')


print('Saving Logistic Regression Model Metrics....')
metrics_lr = {
    'Model': 'Logistic Regression',
    'ROC AUC': roc_auc_score(y_test, y_pred),
    'F1': f1_score(y_test, deposit_decision),
    'Precision': precision_score(y_test, deposit_decision),
    'Recall': recall_score(y_test, deposit_decision),
    'Accuracy': accuracy_score(y_test, deposit_decision)
}

print(metrics_lr)
print()
print('Logistic Regression Model Metrics Saved')
print()


#Random Forest
print('Training Random Forest Classifier....')
print('Identifying Best Performing Parameters')
print()

df_full_train, df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
dicts_train, dicts_val, dicts_test, X_train, X_val, X_test = transform_data(df)

scores = []
for d in [4, 5, 6, 7, 10, 15, 20, None]:
    for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
        rf = RandomForestClassifier(max_depth=d, 
                                    min_samples_leaf=s, 
                                    n_jobs=-1, 
                                    random_state=1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        
        scores.append((d, s, auc))

columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)
best_score = df_scores.iloc[df_scores['auc'].idxmax()]
max_depth = best_score[0].astype(int)
min_samples_leaf = best_score[1].astype(int)

print(f'Optimal max_depth = {max_depth}')
print(f'Optimal min_samples_leaf = {min_samples_leaf}')

print('Indentifying Optimal Random Forest Threshold')
model = RandomForestClassifier(max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            n_jobs = -1,
                            random_state=1)

dv = train(df_train, y_train, model)
y_pred = predict(df_val, dv, model)

print(identify_thresholds(y_val, y_pred))


#Training Final Random Forest Model
print('Training Final Random Forest Model')
print()
df_full_train, df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
dicts_train, dicts_val, dicts_test, X_train, X_val, X_test = transform_data(df)
y_full_train = df_full_train['y'].values
df_full_train = df_full_train.drop(columns = ['y'])

dv = train(df_full_train, y_full_train, model)
y_pred = predict(df_test, dv, model)
deposit_decision = (y_pred >= 0.21).astype(int)
auc = roc_auc_score(y_test, y_pred)
print(f'Auc of the final Random Forest Model = {auc}')


print('Saving Random Forest Model Metrics....')
metrics_rf = {
    'Model': 'Random Forest',
    'ROC AUC': roc_auc_score(y_test, y_pred),
    'F1': f1_score(y_test, deposit_decision),
    'Precision': precision_score(y_test, deposit_decision),
    'Recall': recall_score(y_test, deposit_decision),
    'Accuracy': accuracy_score(y_test, deposit_decision)
}

print(metrics_rf)
print()
print('Random Forest Model Metrics Saved')
print()


#XGBoost

print('Training XGBoost Classifier....')
df_full_train, df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
dicts_train, dicts_val, dicts_test, X_train, X_val, X_test = transform_data(df)

y_full_train = df_full_train['y'].values
dicts_full_train = df_full_train.drop(columns=['y']).to_dict(orient='records')

dv_full = DictVectorizer(sparse=False)
X_full_train = dv_full.fit_transform(dicts_full_train)


print('Identifying Optimal Parameters via Grid Search with Cross Validation')
print()
#Defining grid of parameters
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],    # eta
    'max_depth':      [3, 5, 7, 10],
    'min_child_weight':[1, 3, 5, 10]
}

xgb = XGBClassifier(
    eval_metric='logloss',
    scale_pos_weight=(len(y_full_train) - sum(y_full_train)) / sum(y_full_train),  # handle imbalance
    random_state=1
)

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_full_train, y_full_train)
print("Best params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)


print('Identifying Optimal Threshold')
print()
model = grid.best_estimator_
y_pred = predict(df_val, dv_full, model)
best_threshold = identify_thresholds(y_val, y_pred)
print(best_threshold)

print('Training Final XGBoost Model')

y_pred = predict(df_test, dv_full, model)
deposit_decision = (y_pred >= best_threshold['threshold']).astype(int)
auc = roc_auc_score(y_test, y_pred)
print(f'Auc of the final XGBoost Model = {auc}')

print('Saving XGBoost Model Metrics....')

metrics_xgb = {
    'Model': 'XGBoost',
    'ROC AUC': roc_auc_score(y_test, y_pred),
    'F1': f1_score(y_test, deposit_decision),
    'Precision': precision_score(y_test, deposit_decision),
    'Recall': recall_score(y_test, deposit_decision),
    'Accuracy': accuracy_score(y_test, deposit_decision)
}

print(metrics_xgb)
print()
print('XGBoost Model Metrics Saved')
print()


print("Model Comparison:")
results_df = pd.DataFrame([metrics_xgb, metrics_rf, metrics_lr])
print(results_df.round(5))

# Quick winner summary
best_auc = results_df.loc[results_df['ROC AUC'].idxmax(), 'Model']
best_f1 = results_df.loc[results_df['F1'].idxmax(), 'Model']

print(f"\nModel with Best AUC: {best_auc}")
print(f"Model with Best F1: {best_f1}")



print('XGBoost is the Best Classifier for this Dataset:')
print('Features in Dataset have relatively low individual correlations with the target variable')
print('XGBoost excels at finding feature interactions and non-linear patterns')
print()


#Feature Importance - XGBoost
feature_names = dv_full.feature_names_
importance_scores = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance_scores
}).sort_values('importance', ascending=False)

feature_importance_df = feature_importance_df.reset_index(drop=True)

print("Top 5 Features by Importance:")
print(feature_importance_df.head(5)[['feature', 'importance']])


# Save the model
model = grid.best_estimator_
dv = dv_full 

output_file = f'XGBoost_model.bin' 

with open(output_file, 'wb') as f_out:        #f_out = file output
    pickle.dump((dv, model), f_out)

print(f'XGBoost Model is saved to {output_file}')

