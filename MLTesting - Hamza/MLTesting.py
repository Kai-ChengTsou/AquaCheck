import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy import signal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KernelDensity
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau

from xgboost import XGBRegressor

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

data['TBW %_BIA'] = data['TBW kg_BIA']/data['weight kg']

data.head()

X = data[['SEX M=1 F=2', 'agechild_years', 'weight kg', 'height_cm', 'waistc_cm', 'Biomipedance Index']]
y = data['TBW kg_DDM']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = (mean_absolute_error(y_test, y_pred))**(1/2)
r2 = r2_score(y_test, y_pred)

print(f'Linear Regression Mean Absolute Error: {mae}')
print(f'R-squared: {r2}\n')

coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print(f"{coefficients}\n")

model = Ridge(alpha=15)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Ridge Mean Absolute Error: {mae}')
print(f'R-squared: {r2}\n')

coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print(f"{coefficients}\n")


model = Lasso(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Lasso Mean Absolute Error: {mae}')
print(f'R-squared: {r2}\n')

coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print(f"{coefficients}\n")


model = ElasticNet(alpha=1.0, l1_ratio=0.5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'ElasticNet Mean Absolute Error: {mae}')
print(f'R-squared: {r2}\n')

coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print(f"{coefficients}\n")

X = data[['weight kg', 'height_cm', 'waistc_cm', 'Biomipedance Index']]
y = data['TBW kg_DDM']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Ridge 1.0 Mean Absolute Error: {mae}')
print(f'R-squared: {r2}\n')

coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print(f"{coefficients}\n")

# Assuming 'data' is your DataFrame
X = data[['SEX M=1 F=2', 'agechild_years', 'weight kg', 'height_cm', 'waistc_cm', 'Biomipedance Index']]
y = data['TBW kg_DDM']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a simple neural network model
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer for regression task

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, validation_data=(X_test_scaled, y_test))

# Evaluate the model
loss = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error on Test Data: {loss}')

# Make predictions
predictions = model.predict(X_test_scaled)

# Assuming 'data' is your DataFrame
X = data[['SEX M=1 F=2', 'agechild_years', 'weight kg', 'height_cm', 'waistc_cm', 'Biomipedance Index']]
y = data['TBW kg_DDM']

model_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model_nn.compile(optimizer='adam', loss='mean_squared_error')
model_nn.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
predictions_nn = model_nn.predict(X_test)
mse_nn = mean_absolute_error(y_test, predictions_nn)

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # 3D projection

# Scatter plot of Biomipedance Index, Total Body Water, and Height
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], y_test, label='Actual', color='blue', alpha=0.6)
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], predictions_nn, label='Predicted', color='red', alpha=0.6)

# Set labels for each axis
ax.set_xlabel('Biomipedance Index')
ax.set_ylabel('Height (cm)')
ax.set_zlabel('Total Body Water')

# Add a legend with MSE
ax.legend(labels=[f'MAE: {mse_nn:.2f}'], loc='upper right')

plt.title("Total Body Water vs. Biomipedance Index and Height : NN")
plt.show()

# Define the model using the Functional API
input_layer = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(input_layer)
x = Dense(32, activation='relu')(x)
output_layer = Dense(1)(x)

model_nn = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model_nn.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
history = model_nn.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test))
# Predict on the test set
predictions_nn = model_nn.predict(X_test).flatten()  # Flatten the predictions array

# Calculate Mean Absolute Error
mse_nn = mean_absolute_error(y_test, predictions_nn)

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Adjust the plotting based on your DataFrame's structure
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], y_test, label='Actual', color='blue', alpha=0.6)
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], predictions_nn, label='Predicted', color='red', alpha=0.6)

# Set labels for each axis
ax.set_xlabel('Biomipedance Index')
ax.set_ylabel('Height (cm)')
ax.set_zlabel('Total Body Water')

# Add a legend with MAE
ax.legend(title=f'MAE: {mse_nn:.2f}', loc='upper right')

plt.title("Total Body Water vs. Biomipedance Index and Height : Absolute ERROR NN")
plt.show()

# Train a Random Forest regressor
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # No need for n_estimators
predictions_lr = lr_model.predict(X_test)

predictions_nn = predictions_nn.reshape(-1)
combined_predictions = (predictions_nn + predictions_lr) / 2

mse_combined = mean_absolute_error(y_test, combined_predictions)
print(f"Mean Absolute Error (Combined): {mse_combined}")

# Adjust the plotting to show combined predictions
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], y_test, label='Actual', color='blue', alpha=0.6)
# Neural Network Predictions
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], predictions_nn, label='NN Predicted', color='red', alpha=0.6)
# Combined Predictions
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], combined_predictions, label='Combined Predicted', color='green', alpha=0.6)

# Set labels for each axis
ax.set_xlabel('Biomipedance Index')
ax.set_ylabel('Height (cm)')
ax.set_zlabel('Total Body Water')

# Add a legend with MAE
ax.legend(title=f'MAE: {mse_combined:.2f}', loc='upper right')

plt.title("Total Body Water vs. Biomipedance Index and Height : Combined Model")
plt.show()

# Train a Random Forest regressor
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # No need for n_estimators
predictions_lr = lr_model.predict(X_test)

mse_linear = mean_absolute_error(y_test, predictions_lr)
print(f"Mean Absolute Error (Combined): {mse_linear}")

# Adjust the plotting to show combined predictions
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], y_test, label='Actual', color='blue', alpha=0.6)
# Neural Network Predictions
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], predictions_lr, label='NN Predicted', color='red', alpha=0.6)
# Set labels for each axis
ax.set_xlabel('Biomipedance Index')
ax.set_ylabel('Height (cm)')
ax.set_zlabel('Total Body Water')

# Add a legend with MAE
ax.legend(title=f'MAE: {mse_linear:.2f}', loc='upper right')

plt.title("Total Body Water vs. Biomipedance Index and Height : Linear Model (All parameters)")
plt.show()

data['Bioimpedance Index'] = (data['height_cm']**2)/data['Impedance Z50 kHz']

mean_bia = data['Bioimpedance Index'].mean()
std_bia = data['Bioimpedance Index'].std()

# Calculate 'BMI'
data['BMI'] = data['weight kg'] / (data['height_cm'] / 100)**2

# Calculate 'BSA'
data['BSA'] = 0.007184 * data['height_cm']**0.725 * data['weight kg']**0.425

# Calculate 'Height to Weight' ratio
data['Height to Weight'] = data['height_cm'] / data['weight kg']

# Calculate 'Height to Waist' ratio
# Assuming 'waistc_cm' is the correct column name for waist circumference
data['Height to Waist'] = data['height_cm'] / data['waistc_cm']

# Calculate 'Weight to Waist' ratio
data['Weight to Waist'] = data['weight kg'] / data['waistc_cm']

# Calculate 'ABSI'
# Assuming 'waistc_cm' is the correct column name for waist circumference
data['ABSI'] = data['waistc_cm'] * (data['BMI']) / data['height_cm']

# Define features for linear regression
# Ensure the feature names match exactly with the column names in your dataset
x = data[['weight kg', 'BMI', 'BSA', 'Height to Weight', 'Bioimpedance Index']]
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

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual values
ax.scatter(x_test['Bioimpedance Index'], x_test['BSA'], y_test, label='Actual', color='blue', alpha=0.6)
ax.scatter(x_test['Bioimpedance Index'], x_test['BSA'], y_pred, label='Predicted', color='red', alpha=0.6)

# Set labels for axes
ax.set_xlabel('Bioimpedance Index')
ax.set_ylabel('BSA')
ax.set_zlabel('Total Body Water (TBW kg_DDM)')

# Add a legend with MAE
ax.legend(title=f'MAE: {mae:.2f}', loc='upper right')

plt.title("Total Body Water vs. Bioimpedance Index and Height: Linear Model")
plt.show()

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

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual values
ax.scatter(x_test['Bioimpedance Index'], x_test['BSA'], y_test, label='Actual', color='blue', alpha=0.6)

# Neural Network Predictions
ax.scatter(x_test['Bioimpedance Index'], x_test['BSA'], nn_predictions, label='Linear NN Predicted', color='green', alpha=0.6)

# Set labels for axes
ax.set_xlabel('Bioimpedance Index')
ax.set_ylabel('BSA')
ax.set_zlabel('Total Body Water (TBW kg_DDM)')

# Add a legend with MAE
ax.legend(title=f'NN MAE: {nn_mae:.2f}', loc='upper right')

plt.title("Total Body Water vs. Bioimpedance Index and BSA: Linear Neural Network Model")
plt.show()

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

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual values
ax.scatter(x_test['Bioimpedance Index'], x_test['BSA'], y_test, label='Actual', color='blue', alpha=0.6)

# Neural Network Predictions
ax.scatter(x_test['Bioimpedance Index'], x_test['BSA'], nn_predictions, label='Linear DNN Predicted', color='green', alpha=0.6)

# Set labels for axes
ax.set_xlabel('Bioimpedance Index')
ax.set_ylabel('BSA')
ax.set_zlabel('Total Body Water (TBW kg_DDM)')

# Add a legend with MAE
ax.legend(title=f'NN MAE: {nn_mae:.2f}', loc='upper right')

plt.title("Total Body Water vs. Bioimpedance Index and BSA: Linear Deep Neural Network Model")
plt.show()

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

# Plotting the ensemble predictions
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual values
ax.scatter(x_test['Bioimpedance Index'], x_test['BSA'], y_test, label='Actual', color='blue', alpha=0.6)

# Ensemble Predictions
ax.scatter(x_test['Bioimpedance Index'], x_test['BSA'], ensemble_predictions, label='Ensemble Predicted', color='purple', alpha=0.6)

# Set labels for axes
ax.set_xlabel('Bioimpedance Index')
ax.set_ylabel('BSA')
ax.set_zlabel('Total Body Water (TBW kg_DDM)')

# Add a legend with MAE
ax.legend(title=f'Ensemble MAE: {ensemble_mae:.2f}', loc='upper right')

plt.title("Total Body Water vs. Bioimpedance Index and BSA: Ensemble Model")
plt.show()