import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")

#parameters

C = 1.0
n_splits = 5

#data preparation

df = pd.read_csv('data/Heart_disease_cleveland_new.csv')

df.rename(columns={'cp':'chest_pain_type',
                  'trestbps':'resting_bp',
                  'fbs':'fasting_blood_sugar',
                  'restecg':'resting_ecg',
                   'thalach':'max_hr_achieved',
                  'exang':'exercise_angina',
                  'oldpeak':'st_depression',
                   'slope':'st_slope',
                   'ca':'no_vessels_fluoroscopy',
                   'thal':'thal_result',
                   'target':'heart_disease'
                  }, inplace = True)


numerical = ['age', 'resting_bp', 'chol', 'max_hr_achieved', 'st_depression',
             'no_vessels_fluoroscopy']

categorical = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 
               'exercise_angina', 'st_slope', 'thal_result']

for c in categorical:
    df[c] = df[c].astype("object")


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

#training

def train(df_train, y_train, C=1.0):
    dicts_train = df_train[categorical+numerical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts_train)

    model = LogisticRegression(solver='lbfgs', max_iter = 1000, C=C)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical+numerical].to_dict(orient='records')

    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

#training

print(f'Doing {n_splits}-Fold Validation with C={C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.heart_disease.values
    y_val = df_val.heart_disease.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append((C, auc))

print('Validation Results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


#training the final model

print('Training the final model')

dv, model = train(df_full_train, df_full_train.heart_disease.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.heart_disease.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc of the final model = {auc}')


# Save the model
output_file = f'model_C={C}.bin' 

with open(output_file, 'wb') as f_out:        #f_out = file output
    pickle.dump((dv, model), f_out)

print(f'model is saved to {output_file}')
