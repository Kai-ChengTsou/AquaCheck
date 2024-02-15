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

data = pd.read_excel('./MLTesting - Hamza/Datasets/Senegalese Dataset.xlsx', header=1)
data = data.drop(['Group predic=1_validation=2','ID','waistc_cm','hydration','Hydrated State'], axis=1)

data['Bioimpedance Index'] = (data['height_cm'] ** 3) / data['Impedance Z50 kHz']
data['BMI'] = data['weight kg'] / (data['height_cm'] / 100) ** 2
data['BSA'] = 0.007184 * data['height_cm'] ** 0.725 * data['weight kg'] ** 0.425
data['Height to Weight'] = data['height_cm'] / data['weight kg']
data['Gender'] = 2 - data['SEX M=1 F=2']

data = data.drop('SEX M=1 F=2', axis=1)

# print(data.head())

data.rename(columns={'weight kg': 'Weight', 'height_cm': 'Height', 'agechild_years' : 'Age'}, inplace=True)

x_tbw = data[['Gender','Age','Weight','Height','BMI','BSA','Height to Weight','Bioimpedance Index']]
x_ffm = data[['Gender','Age','Weight','Height','BMI','BSA','Height to Weight','Bioimpedance Index']]
y_tbw = data['TBW kg_DDM']
y_ffm = data['FFM kg_DDM']

def boost_models(X_train, y_train, X_test, y_test, model):
    regr_trans = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='normal'))
    regr_trans.fit(X_train, y_train)
    yhat = regr_trans.predict(X_test)
    algoname = model.__class__.__name__
    return algoname, round(r2_score(y_test, yhat), 3), round(mean_absolute_error(y_test, yhat), 2), round(np.sqrt(mean_squared_error(y_test, yhat)), 2), round(np.mean(y_test - yhat), 2)

x_tbw_train, x_tbw_test, y_tbw_train, y_tbw_test = train_test_split(x_tbw, y_tbw, test_size=0.2, random_state=42)
x_ffm_train, x_ffm_test, y_ffm_train, y_ffm_test = train_test_split(x_ffm, y_ffm, test_size=0.2, random_state=42)

models = [GradientBoostingRegressor(), lgb.LGBMRegressor(), xgb.XGBRegressor(), LinearRegression(), Lasso(), ElasticNet(), KNeighborsRegressor(), DecisionTreeRegressor(), CatBoostRegressor()]

tbw_model = LinearRegression()
# tbw_model.fit(x_tbw_train, y_tbw_train)
tbw_model.fit(x_tbw, y_tbw)

ffm_model = LinearRegression()
# ffm_model.fit(x_ffm_train, y_ffm_train)
ffm_model.fit(x_ffm, y_ffm)

y_tbw_pred = tbw_model.predict(x_tbw_test)
y_ffm_pred = ffm_model.predict(x_ffm_test)


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
    print(f'RMSE:      {rmse:.4f}')
    print(f'R-squared: {r2:.4f}')
    print(f'Bias:      {bias:.4f}\n')

base_models = [
    ('linear_regression', LinearRegression()),
    ('lgbm', lgb.LGBMRegressor()),
    ('xgb', xgb.XGBRegressor())
]

meta_model = LinearRegression()

stacking_model_tbw = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_model_tbw.fit(x_tbw, y_tbw)
stacking_pred_tbw = stacking_model_tbw.predict(x_tbw_test)

stacking_model_ffm = StackingRegressor(estimators=base_models, final_estimator=meta_model)
stacking_model_ffm.fit(x_ffm, y_ffm)
stacking_pred_ffm = stacking_model_ffm.predict(x_ffm_test)

tbw_scores = evaluate_models(x_tbw, y_tbw, models)
ffm_scores = evaluate_models(x_ffm, y_ffm, models)

print("\nTBW Model Scores:")
print(tbw_scores)
print("\nFFM Model Scores:")
print(ffm_scores)

print("\nLinear model Metrics")
evaluation_metrics(y_tbw_test, y_tbw_pred)
evaluation_metrics(y_ffm_test, y_ffm_pred)

print("\nStacking Ensemble Metrics")
evaluation_metrics(y_tbw_test, stacking_pred_tbw)
evaluation_metrics(y_ffm_test, stacking_pred_ffm)

tests = ['8AM', '10AM', '12PM', '1230PM', '1PM', '2PM', '4PM', '6PM']
data = []
data_I = []
data_Q = []

for time in tests:
  test_data = pd.read_csv('./MLTesting - Hamza/Datasets/' + time + ' Standard.csv')[10:-4]
  data.append(test_data)
  temp_data_I = test_data.iloc[:,2].values
  temp_data_Q = test_data.iloc[:,3].values

  data_I.append(temp_data_I.astype(float) * 0.016 + 14)
  data_Q.append(temp_data_Q.astype(float) * 0.016 + 14)

R_post = []
Xc_post = []

for i in range(len(data_I)):
  b, a = signal.butter(3, 0.02)
  data_I[i] = signal.filtfilt(b, a, data_I[i])
  data_Q[i] = signal.filtfilt(b, a, data_Q[i])

for i in range(len(data_I)):
  kdeI = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_I[i].reshape(-1,1))
  densityI = kdeI.score_samples(data_I[i].reshape(-1,1))

  kdeQ = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_Q[i].reshape(-1,1))
  densityQ = kdeQ.score_samples(data_Q[i].reshape(-1,1))

  R_post.append(data_I[i][max(enumerate(densityI),key=lambda x: x[1])[0]])
  Xc_post.append(data_Q[i][max(enumerate(densityQ),key=lambda x: x[1])[0]])

Imp_post = (np.square(Xc_post) + np.square(R_post))**(1/2)
Pha_post = np.rad2deg(np.arctan2(Xc_post, R_post))

weight_at_test = [67.5,67.7,67.4,68,68.7,68.8,69.2,68.9]
ffm_percent_at_test = [86.6,85.4,84.7,84.8,84.2,84.1,83.9,85]
tbw_percent_at_test = [63.2,62.4,61.8,61.9,61.5,61.4,61.3,62]
tbw_kg_at_test = []
ffm_kg_at_test = []

tbw_from_equation = []
ffm_from_equation = []
for i in range(len(weight_at_test)):
  tbw_kg_at_test.append(weight_at_test[i]*tbw_percent_at_test[i]/100)
  ffm_kg_at_test.append(weight_at_test[i]*ffm_percent_at_test[i]/100)
  tbw_from_equation.append(1.2 + 0.45 * 173**2 / R_post[i] + 0.18 * 70)
  ffm_from_equation.append(-10.68 + 0.65*173**2/R_post[i] + 0.26*70 + 0.02*R_post[i])

dehydrated_test_data = pd.DataFrame({
  'Gender': [1]*8, 
  'Weight': [70]*8,
  'Height': [173]*8, 
  'Age': [22.52]*8,
  'Impedance': Imp_post})

dehydrated_test_data['Bioimpedance Index'] = (dehydrated_test_data['Height'] ** 3) / dehydrated_test_data['Impedance']
dehydrated_test_data['BMI'] = dehydrated_test_data['Weight'] / (dehydrated_test_data['Height'] / 100) ** 2
dehydrated_test_data['BSA'] = 0.007184 * dehydrated_test_data['Height'] ** 0.725 * dehydrated_test_data['Weight'] ** 0.425
dehydrated_test_data['Height to Weight'] = dehydrated_test_data['Height'] / dehydrated_test_data['Weight']

x_tbw_dehydrated = dehydrated_test_data[['Gender','Age','Weight','Height','BMI','BSA','Height to Weight','Bioimpedance Index']]
x_ffm_dehydrated = dehydrated_test_data[['Gender','Age','Weight','Height','BMI','BSA','Height to Weight','Bioimpedance Index']]

tbw_pred = tbw_model.predict(x_tbw_dehydrated)
ffm_pred = ffm_model.predict(x_ffm_dehydrated)

tbw_pred_stack = stacking_model_tbw.predict(x_tbw_dehydrated)
ffm_pred_stack = stacking_model_ffm.predict(x_ffm_dehydrated)

plt.figure(figsize=(12, 5))
time_of_day = [8,10,12,12.5,13,14,16,18]
# Plot for TBW
plt.subplot(1, 2, 1)
plt.plot(time_of_day, tbw_kg_at_test, label='Ground Truth')
plt.plot(time_of_day, tbw_from_equation +np.average(tbw_kg_at_test)-np.average(tbw_from_equation), label='Equation Prediction')
plt.plot(time_of_day, tbw_pred +np.average(tbw_kg_at_test)-np.average(tbw_pred), label='Linear Model Prediction')
plt.plot(time_of_day, tbw_pred_stack +np.average(tbw_kg_at_test)-np.average(tbw_pred_stack), label='Model Prediction')
plt.title('Dehydrated Test - TBW')
plt.ylabel('TBW (kg)')
plt.xlabel('Time of day (h)')
plt.legend()

# Plot for FFM
plt.subplot(1, 2, 2)
plt.plot(time_of_day, ffm_kg_at_test, label='Ground Truth')
plt.plot(time_of_day, ffm_from_equation +np.average(ffm_kg_at_test)-np.average(ffm_from_equation), label='Equation Prediction')
plt.plot(time_of_day, ffm_pred +np.average(ffm_kg_at_test)-np.average(ffm_pred), label='Linear Model Prediction')
plt.plot(time_of_day, ffm_pred_stack +np.average(ffm_kg_at_test)-np.average(ffm_pred_stack), label='Model Prediction')
plt.title('Dehydrated Test - FFM')
plt.ylabel('FFM (kg)')
plt.xlabel('Time of day (h)')
plt.legend()
plt.show()

#####################################################################################################################

collected_data = pd.read_excel('./MLTesting - Hamza/Datasets/Collected Dataset.xlsx')
collected_data['Index'] = collected_data['Height'] **3 / collected_data['Imp']
x = collected_data[['R','Xc','Weight','Height','Age']]
y_tbw = collected_data['TBW kg_Scale']
y_ffm = collected_data['FFM kg_Sale']

filtered_data = collected_data[collected_data['Person'] != 'Luis']
x_train = filtered_data[['Index','Xc','Weight','Height','Age']]

y_tbw_train = filtered_data['TBW kg_Scale']
y_ffm_train = filtered_data['FFM kg_Sale']

tbw_model_col = LinearRegression()
ffm_model_col = LinearRegression()
tbw_model_col.fit(x_train,y_tbw_train)
ffm_model_col.fit(x_train,y_ffm_train)

filtered_data = collected_data[collected_data['Person'] == 'Luis']
x = filtered_data[['Index','Xc','Weight','Height','Age']]

pred_tbw = tbw_model_col.predict(x).flatten()
pred_ffm = ffm_model_col.predict(x).flatten()

plt.figure(figsize=(12, 5))
# Plot for TBW
plt.subplot(1, 2, 1)
plt.plot(time_of_day, filtered_data['TBW kg_Scale'], label='Ground Truth')
plt.plot(time_of_day, pred_tbw, label='Prediction')
plt.title('Dehydrated Test - TBW')
plt.ylabel('TBW (kg)')
plt.xlabel('Time of day (h)')
plt.legend()

# Plot for FFM
plt.subplot(1, 2, 2)
plt.plot(time_of_day, filtered_data['FFM kg_Sale'], label='Ground Truth')
plt.plot(time_of_day, pred_ffm, label='Prediction')
plt.title('Dehydrated Test - FFM')
plt.ylabel('FFM (kg)')
plt.xlabel('Time of day (h)')
plt.legend()
plt.show()