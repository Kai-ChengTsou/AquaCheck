import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from scipy import signal
from itertools import combinations

from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

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

x_tbw_train, x_tbw_test, y_tbw_train, y_tbw_test = train_test_split(x_tbw, y_tbw, test_size=0.2, random_state=42)
x_ffm_train, x_ffm_test, y_ffm_train, y_ffm_test = train_test_split(x_ffm, y_ffm, test_size=0.2, random_state=42)

tbw_model = LinearRegression()
# tbw_model.fit(x_tbw_train, y_tbw_train)
tbw_model.fit(x_tbw, y_tbw)

ffm_model = LinearRegression()
# ffm_model.fit(x_ffm_train, y_ffm_train)
ffm_model.fit(x_ffm, y_ffm)

y_tbw_pred = tbw_model.predict(x_tbw_test)
y_ffm_pred = ffm_model.predict(x_ffm_test)

def evaluation_metrics (y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred))**(1/2)
    r2 = r2_score(y_test, y_pred)
    bias = np.mean(y_test - y_pred)
    print(f'MAE:       {mae:.4f}')
    print(f'RMSE:      {rmse:.4f}')
    print(f'R-squared: {r2:.4f}')
    print(f'Bias:      {bias:.4f}\n')

evaluation_metrics(y_tbw_test, y_tbw_pred)
evaluation_metrics(y_ffm_test, y_ffm_pred)

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
  'Imp': Imp_post,
  'R': R_post,
  'Xc':Xc_post,
  'Pha': Pha_post})

dehydrated_test_data['Bioimpedance Index'] = (dehydrated_test_data['Height'] ** 3) / dehydrated_test_data['Imp']
dehydrated_test_data['BMI'] = dehydrated_test_data['Weight'] / (dehydrated_test_data['Height'] / 100) ** 2
dehydrated_test_data['BSA'] = 0.007184 * dehydrated_test_data['Height'] ** 0.725 * dehydrated_test_data['Weight'] ** 0.425
dehydrated_test_data['Height to Weight'] = dehydrated_test_data['Height'] / dehydrated_test_data['Weight']

x_tbw_dehydrated = dehydrated_test_data[['Gender','Age','Weight','Height','BMI','BSA','Height to Weight','Bioimpedance Index']]
x_ffm_dehydrated = dehydrated_test_data[['Gender','Age','Weight','Height','BMI','BSA','Height to Weight','Bioimpedance Index']]

tbw_pred = tbw_model.predict(x_tbw_dehydrated)
ffm_pred = ffm_model.predict(x_ffm_dehydrated)

plt.figure(figsize=(12, 5))
time_of_day = [8,10,12,12.5,13,14,16,18]
# Plot for TBW
plt.subplot(1, 2, 1)
plt.plot(time_of_day, tbw_kg_at_test, label='Ground Truth')
plt.plot(time_of_day, tbw_from_equation, label='Equation Prediction')
# plt.plot(time_of_day, tbw_pred, label='Model Prediction')
plt.title('Dehydrated Test - TBW')
plt.ylabel('TBW (kg)')
plt.xlabel('Time of day (h)')
plt.legend()

# Plot for FFM
plt.subplot(1, 2, 2)
plt.plot(time_of_day, ffm_kg_at_test, label='Ground Truth')
plt.plot(time_of_day, ffm_from_equation, label='Equation Prediction')
# plt.plot(time_of_day, ffm_pred, label='Model Prediction')
plt.title('Dehydrated Test - FFM')
plt.ylabel('FFM (kg)')
plt.xlabel('Time of day (h)')
plt.legend()
plt.show()

#####################################################################################################################
collected_data = pd.read_excel('./MLTesting - Hamza/Datasets/Collected Dataset.xlsx')
collected_data['Bioimpedance Index'] = collected_data['Height'] **3 / collected_data['Imp']
collected_data['BMI'] = collected_data['Weight'] / (collected_data['Height'] / 100) ** 2
collected_data['BSA'] = 0.007184 * collected_data['Height'] ** 0.725 * collected_data['Weight'] ** 0.425
collected_data['Height to Weight'] = collected_data['Height'] / collected_data['Weight']

filtered_data = collected_data[collected_data['Imp'] < 480]
x_tbw = filtered_data[['Xc', 'BMI', 'Height', 'Age', 'BMI', 'BSA', 'Height to Weight', 'Pha']]
x_ffm = filtered_data[['Xc', 'BMI', 'Height', 'Age', 'BMI', 'Height to Weight', 'Pha']]
y_tbw = filtered_data['TBW kg_Scale']
y_ffm = filtered_data['FFM kg_Sale']

tbw_model_col = LinearRegression()
ffm_model_col = LinearRegression()
tbw_model_col.fit(x_tbw,y_tbw)
ffm_model_col.fit(x_ffm,y_ffm)

filtered_test_data = collected_data[collected_data['Person'] == 'Luis']
x_tbw_test = dehydrated_test_data[['Xc', 'BMI', 'Height', 'Age', 'BMI', 'BSA', 'Height to Weight', 'Pha']]
x_ffm_test = dehydrated_test_data[['Xc', 'BMI', 'Height', 'Age', 'BMI', 'Height to Weight', 'Pha']]

pred_tbw = tbw_model_col.predict(x_tbw_test)
pred_ffm = ffm_model_col.predict(x_ffm_test)

plt.figure(figsize=(12, 5))
# Plot for TBW
plt.subplot(1, 2, 1)
plt.plot(time_of_day, filtered_test_data['TBW kg_Scale'], label='Ground Truth')
plt.plot(time_of_day, tbw_from_equation, label='Equation Prediction')
plt.plot(time_of_day, pred_tbw, label='Prediction')
plt.title('Dehydrated Test - TBW')
plt.ylabel('TBW (kg)')
plt.xlabel('Time of day (h)')
plt.legend()

# Plot for FFM
plt.subplot(1, 2, 2)
plt.plot(time_of_day, filtered_test_data['FFM kg_Sale'], label='Ground Truth')
plt.plot(time_of_day, ffm_from_equation, label='Equation Prediction')
plt.plot(time_of_day, pred_ffm, label='Model Prediction')
plt.title('Dehydrated Test - FFM')
plt.ylabel('FFM (kg)')
plt.xlabel('Time of day (h)')
plt.legend()
plt.show()



# tbw_train_pred = tbw_model.predict(x_tbw_train)  
# tbw_test_pred = tbw_model.predict(x_tbw_test)    
# ffm_train_pred = ffm_model.predict(x_ffm_train)  
# ffm_test_pred = ffm_model.predict(x_ffm_test)   
# x_tbw_train['LR Pred'] = tbw_train_pred
# x_tbw_test['LR Pred'] = tbw_test_pred
# x_ffm_train['LR Pred'] = ffm_train_pred
# x_ffm_test['LR Pred'] = ffm_test_pred

# tbw_scaler = StandardScaler()
# ffm_scaler = StandardScaler()
# x_tbw_train_scaled = tbw_scaler.fit_transform(x_tbw_train)
# x_tbw_test_scaled = tbw_scaler.transform(x_tbw_test)
# x_ffm_train_scaled = ffm_scaler.fit_transform(x_ffm_train)
# x_ffm_test_scaled = ffm_scaler.transform(x_ffm_test)

# # Redefine and compile a new model with regularization
# tbw_input_layer = Input(shape=(x_tbw_train.shape[1]))
# tbw_model_layer = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(tbw_input_layer)
# tbw_model_layer = Dropout(0.3)(tbw_model_layer)
# tbw_model_layer = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(tbw_model_layer)
# tbw_model_layer = Dropout(0.3)(tbw_model_layer)
# tbw_model_layer = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(tbw_model_layer)
# tbw_model_layer = Dropout(0.3)(tbw_model_layer)
# tbw_model_layer = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(tbw_model_layer)
# tbw_output_layer = Dense(1)(tbw_model_layer)

# ffm_input_layer = Input(shape=(x_ffm_train.shape[1]))
# ffm_model_layer = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(ffm_input_layer)
# ffm_model_layer = Dropout(0.3)(ffm_model_layer)
# ffm_model_layer = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(ffm_model_layer)
# ffm_model_layer = Dropout(0.3)(ffm_model_layer)
# ffm_model_layer = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(ffm_model_layer)
# ffm_model_layer = Dropout(0.3)(ffm_model_layer)
# ffm_model_layer = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(ffm_model_layer)
# ffm_output_layer = Dense(1)(ffm_model_layer)

# tbw_model_nn = Model(inputs=tbw_input_layer, outputs=tbw_output_layer)
# tbw_model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
# ffm_model_nn = Model(inputs=ffm_input_layer, outputs=ffm_output_layer)
# ffm_model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')

# tbw_model_nn.fit(
#     x_tbw_train_scaled, y_tbw_train,
#     validation_data=(x_tbw_test_scaled, y_tbw_test),
#     epochs=100,
#     batch_size=16)
# ffm_model_nn.fit(
#     x_ffm_train_scaled, y_ffm_train,
#     validation_data=(x_ffm_test_scaled, y_ffm_test),
#     epochs=100,
#     batch_size=16)

# tbw_nn_predictions = tbw_model_nn.predict(x_tbw_test_scaled).flatten()
# ffm_nn_predictions = tbw_model_nn.predict(x_ffm_test_scaled).flatten()

# evaluation_metrics(y_tbw_test, tbw_nn_predictions)
# evaluation_metrics(y_ffm_test, ffm_nn_predictions)