import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt
from scipy import signal
import gspread

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

data = pd.read_excel('./Datasets/Senegalese Dataset.xlsx', header=1)
data = data.drop('Group predic=1_validation=2', axis=1)
data = data.drop('ID', axis=1)
data = data.drop('FFM kg_BIA', axis=1)
data = data.drop('FM kg_BIA', axis=1)
data = data.drop('%BF_BIA', axis=1)
data = data.drop('%BF_DDM', axis=1)
data = data.drop('FFM kg_DDM', axis=1)
data = data.drop('FM kg_DDM ', axis=1)
data = data.drop('%FFM_DDM', axis=1)
data = data.drop('hydration', axis=1)

# Calculating new features as per instructions
data['TBW %_BIA'] = data['TBW kg_BIA'] / data['weight kg']
data['Bioimpedance Index'] = (data['height_cm'] ** 2) / data['Impedance Z50 kHz']
data['BMI'] = data['weight kg'] / (data['height_cm'] / 100) ** 2
data['BSA'] = 0.007184 * data['height_cm'] ** 0.725 * data['weight kg'] ** 0.425
data['Height to Weight'] = data['height_cm'] / data['weight kg']

# Renaming columns to ensure consistency
data.rename(columns={'weight kg': 'Weight', 'height_cm': 'Height', 'agechild_years' : 'Age'}, inplace=True)

# Defining features and target variable for regression
X = data[['Weight', 'BMI', 'BSA', 'Height to Weight', 'Bioimpedance Index']]
y = data['TBW kg_DDM']

data['Bioimpedance Index'] = (data['Height']**2)/data['Impedance Z50 kHz']

mean_bia = data['Bioimpedance Index'].mean()
std_bia = data['Bioimpedance Index'].std()

# Calculate 'BMI'
data['BMI'] = data['Weight'] / (data['Height'] / 100)**2

# Calculate 'BSA'
data['BSA'] = 0.007184 * data['Height']**0.725 * data['Weight']**0.425

# Calculate 'Height to Weight' ratio
data['Height to Weight'] = data['Height'] / data['Weight']

# Define features for linear regression
# Ensure the feature names match exactly with the column names in your dataset
x = data[['Weight', 'BMI', 'BSA', 'Height to Weight', 'Bioimpedance Index', 'Age', 'Height']]
y = data['TBW kg_DDM']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict on the test set
y_pred = model.predict(x_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = (mean_squared_error(y_test, y_pred))**(1/2)
r2 = r2_score(y_test, y_pred)
bias = np.mean(y_test - y_pred)

# Print the metrics
print(f'MAE:       {mae:.4f}')
print(f'RMSE:      {rmse:.4f}')
print(f'R-squared: {r2:.4f}')
print(f'Bias:      {bias:.4f}\n')

train_preds_lr = model.predict(x_train)  # Predictions for the training set
test_preds_lr = model.predict(x_test)    # Predictions for the test set

x_train['LR_Preds'] = train_preds_lr
x_test['LR_Preds'] = test_preds_lr

input_shape = x_train.shape[1]  # Number of features including the LR predictions
input_layer = Input(shape=(input_shape,))
x = Dense(64, activation='relu')(input_layer)
x = Dense(32, activation='relu')(x)
output_layer = Dense(1)(x)  # Output layer for regression

model_nn = Model(inputs=input_layer, outputs=output_layer)

# Compile the Neural Network model
model_nn.compile(optimizer=Adam(), loss='mean_absolute_error')

# Train the Neural Network model
history_nn = model_nn.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,
    batch_size=8
)

nn_predictions = model_nn.predict(x_test).flatten()  # Flatten to ensure the shape aligns

# Calculate metrics
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
nn_r2 = r2_score(y_test, nn_predictions)
nn_bias = np.mean(y_test - nn_predictions)

# Print the metrics
print(f'NN MAE:       {nn_mae:.4f}')
print(f'NN RMSE:      {nn_rmse:.4f}')
print(f'NN R-squared: {nn_r2:.4f}')
print(f'NN Bias:      {nn_bias:.4f}\n')

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Redefine and compile a new model with regularization
input_layer = Input(shape=(input_shape,))
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
output_layer = Dense(1)(x)

model_nn = Model(inputs=input_layer, outputs=output_layer)
model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')

# Re-train the model with early stopping and learning rate reduction on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

history_nn = model_nn.fit(
    x_train_scaled, y_train,
    validation_data=(x_test_scaled, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping, lr_reduction]
)

# Predict on the scaled test set
nn_predictions = model_nn.predict(x_test_scaled).flatten()

# Recalculate metrics
nn_mae = mean_absolute_error(y_test, nn_predictions)

# Calculate metrics
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))
nn_r2 = r2_score(y_test, nn_predictions)
nn_bias = np.mean(y_test - nn_predictions)

# Print the metrics
print(f'NN MAE:       {nn_mae:.4f}')
print(f'NN RMSE:      {nn_rmse:.4f}')
print(f'NN R-squared: {nn_r2:.4f}')
print(f'NN Bias:      {nn_bias:.4f}\n')

# Predictions from the neural network model
nn_predictions = model_nn.predict(x_test_scaled).flatten()

# Average the predictions from both models
ensemble_predictions = (test_preds_lr + nn_predictions) / 2

# Calculate metrics for the ensemble
ensemble_mae = mean_absolute_error(y_test, ensemble_predictions)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))
ensemble_r2 = r2_score(y_test, ensemble_predictions)
ensemble_bias = np.mean(y_test - ensemble_predictions)

# Print the metrics for the ensemble
print(f'Ensemble MAE:       {ensemble_mae:.4f}')
print(f'Ensemble RMSE:      {ensemble_rmse:.4f}')
print(f'Ensemble R-squared: {ensemble_r2:.4f}')
print(f'Ensemble Bias:      {ensemble_bias:.4f}\n')

tests = ['8AM', '10AM', '12PM', '1230PM', '1PM', '2PM', '4PM', '6PM']
data = []
data_I = []
data_Q = []

for time in tests:
  test_data = pd.read_csv('./Datasets/' + time + ' Standard.csv')[10:-4]
  data.append(test_data)
  temp_data_I = test_data.iloc[:,2].values
  temp_data_Q = test_data.iloc[:,3].values

  data_I.append(temp_data_I.astype(float) * 0.016 + 14)
  data_Q.append(temp_data_Q.astype(float) * 0.016 + 14)

plt.plot(data_I[0])

R_pre = []
Xc_pre = []
for test in range(len(tests)):
  R_pre.append(np.average(data_I[test]))
  Xc_pre.append(np.average(data_Q[test]))

plt.plot([8,10,12,12.5,13,14,16,18],R_pre)
plt.xlabel("Time")
plt.ylabel("Values")
plt.title("Plot with Time on X-Axis")
plt.show()

R_post = []
Xc_post = []

plt.plot(data_I[0])

for i in range(len(data_I)):
  b, a = signal.butter(3, 0.02)
  data_I[i] = signal.filtfilt(b, a, data_I[i])
  data_Q[i] = signal.filtfilt(b, a, data_Q[i])

plt.plot(data_I[0])

for i in range(len(data_I)):
  kdeI = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_I[i].reshape(-1,1))
  densityI = kdeI.score_samples(data_I[i].reshape(-1,1))

  kdeQ = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_Q[i].reshape(-1,1))
  densityQ = kdeQ.score_samples(data_Q[i].reshape(-1,1))

  R_post.append(data_I[i][max(enumerate(densityI),key=lambda x: x[1])[0]])
  Xc_post.append(data_Q[i][max(enumerate(densityQ),key=lambda x: x[1])[0]])
print(R_pre[0])
print((np.array(R_post)**2+np.array(Xc_post)**2)**(1/2))

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot([8, 10, 12, 12.5, 13, 14, 16, 18], R_post)
axs[0, 0].set_xlabel("Time")
axs[0, 0].set_ylabel("Values")
axs[0, 0].set_title("Resistance")

axs[0, 1].plot([8, 10, 12, 12.5, 13, 14, 16, 18], Xc_post)
axs[0, 1].set_xlabel("Time")
axs[0, 1].set_ylabel("Values")
axs[0, 1].set_title("Reactance")

Imp_post = (np.square(Xc_post) + np.square(R_post))**(1/2)
axs[1, 0].plot([8, 10, 12, 12.5, 13, 14, 16, 18], Imp_post)
axs[1, 0].set_xlabel("Time")
axs[1, 0].set_ylabel("Values")
axs[1, 0].set_title("Impedance")

Pha_post = np.rad2deg(np.arctan2(Xc_post, R_post))
axs[1, 1].plot([8, 10, 12, 12.5, 13, 14, 16, 18], Pha_post)
axs[1, 1].set_xlabel("Time")
axs[1, 1].set_ylabel("Values")
axs[1, 1].set_title("Phase")

plt.tight_layout()

plt.show()

weight_at_test = [67.5,67.7,67.4,68,68.7,68.8,69.2,68.9]
ffm_percent_at_test = [86.6,85.4,84.7,84.8,84.2,84.1,83.9,85]
tbw_percent_at_test = [63.2,62.4,61.8,61.9,61.5,61.4,61.3,62]
tbw_kg_at_test = []
ffm_kg_at_test = []

tbw_from_equation = []
tbw_with_accurate_weight = []
ffm_from_equation = []
ffm_with_accurate_weight = []
for i in range(len(weight_at_test)):
  tbw_kg_at_test.append(weight_at_test[i]*tbw_percent_at_test[i]/100)
  ffm_kg_at_test.append(weight_at_test[i]*ffm_percent_at_test[i]/100)
  tbw_from_equation.append(1.2 + 0.45 * 173**2 / R_post[i] + 0.18 * 70)
  tbw_with_accurate_weight.append(1.2 + 0.45 * 173**2 / R_post[i] + 0.18 * weight_at_test[i])
  ffm_from_equation.append(-10.68 + 0.65*173**2/R_post[i] + 0.26*70 + 0.02*R_post[i])
  ffm_with_accurate_weight.append(-10.68 + 0.65*173**2/R_post[i] + 0.26*weight_at_test[i] + 0.02*R_post[i])


plt.figure(figsize=(12, 5))
time_of_day = [8,10,12,12.5,13,14,16,18]
# Plot for TBW
plt.subplot(1, 2, 1)
plt.plot(time_of_day, tbw_kg_at_test, label='Ground Truth')
plt.plot(time_of_day, tbw_from_equation, label='Predicted')
plt.title('Dehydrated Test - TBW')
plt.ylabel('TBW (kg)')
plt.xlabel('Time of day (h)')
plt.legend()

# Plot for FFM
plt.subplot(1, 2, 2)
plt.plot(time_of_day, ffm_kg_at_test, label='Ground Truth')
plt.plot(time_of_day, ffm_from_equation, label='Predicted')
plt.title('Dehydrated Test - FFM')
plt.ylabel('FFM (kg)')
plt.xlabel('Time of day (h)')
plt.legend()
plt.show()

data = pd.read_excel('./Datasets/Collected Dataset.xlsx')

data = data[data['Person'] == 'Luis']
data.head()
x = data[['Imp','Age','Height','Weight']] #,'Pha','Imp','Gender (F=0,M=1)'
y_tbw = data['TBW kg_Scale']
y_ffm = data['FFM kg_Sale']

data['Bioimpedance Index'] = (data['Height']**2)/data['Imp']

mean_bia = data['Bioimpedance Index'].mean()
std_bia = data['Bioimpedance Index'].std()

# Calculate 'BMI'
data['BMI'] = data['Weight'] / (data['Height'] / 100)**2

# Calculate 'BSA'
data['BSA'] = 0.007184 * data['Height']**0.725 * data['Weight']**0.425

# Calculate 'Height to Weight' ratio
data['Height to Weight'] = data['Height'] / data['Weight']

# Define features for linear regression
# Ensure the feature names match exactly with the column names in your dataset
x = data[['Weight', 'BMI', 'BSA', 'Height to Weight', 'Bioimpedance Index', 'Age', 'Height']]
y_tbw = data['TBW kg_Scale']
y_ffm = data['FFM kg_Sale']

x_train, x_test, y_tbw_train, y_tbw_test = train_test_split(x, y_tbw, test_size=0.2, random_state=42)
x_train, x_test, y_ffm_train, y_ffm_test = train_test_split(x, y_ffm, test_size=0.2, random_state=42)

model_tbw = LinearRegression()
model_tbw.fit(x_train, y_tbw_train)
# model_tbw.fit(x, y_tbw)

model_ffm = LinearRegression()
model_ffm.fit(x_train, y_ffm_train)
# model_ffm.fit(x, y_ffm)

y_tbw_pred = model_tbw.predict(x_test)
y_ffm_pred = model_ffm.predict(x_test)

tbw_pred = model_tbw.predict(x)
ffm_pred = model_ffm.predict(x)

plt.figure(figsize=(12, 5))
time_of_day = [8,10,12,12.5,13,14,16,18]
# Plot for TBW
plt.subplot(1, 2, 1)
plt.plot(time_of_day, tbw_kg_at_test, label='Ground Truth')
plt.plot(time_of_day, tbw_pred, label='Predicted')
plt.title('Dehydrated Test - TBW')
plt.ylabel('TBW (kg)')
plt.xlabel('Time of day (h)')
plt.legend()

# Plot for FFM
plt.subplot(1, 2, 2)
plt.plot(time_of_day, ffm_kg_at_test, label='Ground Truth')
plt.plot(time_of_day, ffm_pred, label='Predicted')
plt.title('Dehydrated Test - FFM')
plt.ylabel('FFM (kg)')
plt.xlabel('Time of day (h)')
plt.legend()
plt.show()

x_train_scaled = x_train
x_test_scaled = x_test

# Neural Network for TBW with regularization
model_tbw_reg = Sequential([
    Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],), kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    Dense(1)
])
model_tbw_reg.compile(optimizer=Adam(), loss='mean_absolute_error')
model_tbw_reg.fit(x_train_scaled, y_tbw_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Neural Network for FFM with regularization
model_ffm_reg = Sequential([
    Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],), kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
    Dense(1)
])
model_ffm_reg.compile(optimizer=Adam(), loss='mean_absolute_error')
model_ffm_reg.fit(x_train_scaled, y_ffm_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Making predictions
tbw_pred = model_tbw_reg.predict(x)
ffm_pred = model_ffm_reg.predict(x)

plt.figure(figsize=(12, 5))

# Plot for TBW
plt.subplot(1, 2, 1)
plt.plot(time_of_day, tbw_kg_at_test, label='Ground Truth')  # Ensure this aligns with your actual data
plt.plot(time_of_day, tbw_pred, label='Predicted')
plt.title('Dehydrated Test - TBW')
plt.ylabel('TBW (kg)')
plt.xlabel('Time of day (h)')
plt.legend()

# Plot for FFM
plt.subplot(1, 2, 2)
plt.plot(time_of_day, ffm_kg_at_test, label='Ground Truth')  # Ensure this aligns with your actual data
plt.plot(time_of_day, ffm_pred, label='Predicted')
plt.title('Dehydrated Test - FFM')
plt.ylabel('FFM (kg)')
plt.xlabel('Time of day (h)')
plt.legend()

plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Predicting with Neural Network
nn_predictions = model_nn.predict(x_test_scaled).flatten()

# Predicting with Linear Regression
lr_predictions = model.predict(x_test)

# Combining predictions - Ensemble
ensemble_predictions = (nn_predictions + lr_predictions) / 2

# Evaluating ensemble predictions
ensemble_mae = mean_absolute_error(y_test, ensemble_predictions)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_predictions))
ensemble_r2 = r2_score(y_test, ensemble_predictions)

# Output ensemble evaluation metrics
print(f"Ensemble MAE: {ensemble_mae:.4f}")
print(f"Ensemble RMSE: {ensemble_rmse:.4f}")
print(f"Ensemble R-squared: {ensemble_r2:.4f}")

# Assuming 'time_of_day' is defined corresponding to 'x_test'
plt.figure(figsize=(12, 6))

# Plot TBW predictions from the neural network
plt.subplot(1, 2, 1)
plt.plot(time_of_day, y_test, 'o-', label='Ground Truth TBW')
plt.plot(time_of_day, nn_predictions, 'o-', label='NN Predicted TBW')
plt.title('TBW Predictions')
plt.xlabel('Time of day')
plt.ylabel('TBW (kg)')
plt.legend()

# Plot FFM predictions from linear regression (or any chosen metric)
plt.subplot(1, 2, 2)
plt.plot(time_of_day, y_test, 'o-', label='Ground Truth TBW')
plt.plot(time_of_day, lr_predictions, 'o-', label='LR Predicted TBW')
plt.title('FFM Predictions')
plt.xlabel('Time of day')
plt.ylabel('TBW (kg)')
plt.legend()

plt.tight_layout()
plt.show()