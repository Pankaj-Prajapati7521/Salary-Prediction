import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor



df = pd.read_csv('jobs_in_data.csv')
print(df.shape)
#print(df.columns)
#print(df.isnull().sum())
print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.shape)

x = df.drop(['job_title', 'salary_in_usd'], axis=1)
y = df['salary_in_usd']

cat_features = x.select_dtypes(include=['object']).columns
num_features = x.select_dtypes(exclude=['object']).columns

#preprocessing
preprocessor = ColumnTransformer(
    transformers = [('cat', OneHotEncoder(handle_unknown = 'ignore'), cat_features)],
    remainder = 'passthrough'
)

#pipeline creation
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators = 500, random_state = 42, n_jobs = -1))
])

#spliting the data into train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2, random_state = 42)

#traing the model
pipeline.fit(xtrain, ytrain)
y_pred = pipeline.predict(xtest)

#evaluating the model
rmse = mean_squared_error(ytest, y_pred)**0.5
r2 = r2_score(ytest, y_pred)
print("RMSE: ", rmse)
print("R2 Score: ", r2)

if not os.path.exists('model.pkl'):
    joblib.dump(pipeline, 'model.pkl')
    print("Model saved as model.pkl")
else:
    print("Model already exists. Skipping save.")
