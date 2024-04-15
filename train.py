import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from scipy import signal
from catboost import CatBoostRegressor

from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

import xgboost as xgb
from joblib import dump

def evaluate_models(x_data, y_data, models):
  # Splitting the dataset
  X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
  scores = []
  for model in models:
      scores.append(boost_models(X_train, y_train, X_test, y_test, model))
  return pd.DataFrame(scores, columns=['Model', 'R2 Score', 'MAE', 'RMSE', 'Bias'])

def evaluation_metrics (y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred))**(1/2)
    r2 = r2_score(y_test, y_pred)
    bias = np.mean(y_test - y_pred)
    print(f'MAE:       {mae:.4f}')
    print(f'MAE:       {mae/np.average(y_test) * 100:.4f}%')
    print(f'RMSE:      {rmse:.4f}')
    print(f'R-squared: {r2:.4f}')
    print(f'Bias:      {bias:.4f}\n')
  
def boost_models(X_train, y_train, X_test, y_test, model):
    regr_trans = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='normal'))
    regr_trans.fit(X_train, y_train)
    yhat = regr_trans.predict(X_test)
    algoname = model.__class__.__name__
    return algoname, round(r2_score(y_test, yhat), 3), round(mean_absolute_error(y_test, yhat), 2), round(np.sqrt(mean_squared_error(y_test, yhat)), 2), round(np.mean(y_test - yhat), 2)

def data_processing():
  # Read the data from Excel file
  data = pd.read_excel('.//MLTesting/Datasets/BIAWorldData.xlsx')

  # Drop the specified columns
  data = data.drop(['DataBase', 'Code', 'Region', 'Country', 'HealthStat', 'AthleticStat', 'Etnicity', 'BMI_C', 'BIA_Equip', 'Dil_ICW', 'Dil_ECW'], axis=1)

  data['BioImpedance Index'] = (data['Height'] ** 3) / data['Z']
  data['BSA'] = 0.007184 * data['Height'] ** 0.725 * data['Weight'] ** 0.425
  data['Height to Weight'] = data['Height'] / data['Weight']

  x_tbw = data[['Gender', 'Age', 'Weight', 'Height', 'BMI', 'BSA', 'Height to Weight', 'BioImpedance Index']]
  y_tbw = data['Dil_TBW']

  print(x_tbw)
  print(y_tbw)

  return x_tbw, y_tbw

def train_model():
  x_tbw, y_tbw = data_processing()

  x_tbw_train, x_tbw_test, y_tbw_train, y_tbw_test = train_test_split(x_tbw, y_tbw, test_size=0.2, random_state=42)

  models = [GradientBoostingRegressor(), lgb.LGBMRegressor(), xgb.XGBRegressor(), LinearRegression(), Lasso(), ElasticNet(), KNeighborsRegressor(), DecisionTreeRegressor(), CatBoostRegressor()]

  tbw_model = LinearRegression()
  tbw_model.fit(x_tbw, y_tbw)
  y_tbw_pred = tbw_model.predict(x_tbw_test)

  base_models = [
      ('linear_regression', LinearRegression()),
      ('lgbm', lgb.LGBMRegressor()),
      ('xgb', xgb.XGBRegressor())
  ]

  meta_model = LinearRegression()

  stacking_model_tbw = StackingRegressor(estimators=base_models, final_estimator=meta_model)
  stacking_model_tbw.fit(x_tbw, y_tbw)
  stacking_pred_tbw = stacking_model_tbw.predict(x_tbw_test)

  tbw_scores = evaluate_models(x_tbw, y_tbw, models)

  print("\nTBW Model Scores:")
  print(tbw_scores)

  print("\nLinear model Metrics")
  evaluation_metrics(y_tbw_test, y_tbw_pred)

  print("\nStacking Ensemble Metrics")
  evaluation_metrics(y_tbw_test, stacking_pred_tbw)

  # Save your model
  dump(stacking_model_tbw, 'stacking_model_tbw.joblib')

if __name__ == "__main__":
  train_model()