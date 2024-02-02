import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras

from xgboost import XGBRegressor
from scipy import stats

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# # Define a function to remove outliers using z-scores
# def remove_outliers(df, target, threshold=3):
#     z_scores = np.abs(stats.zscore(df))
#     no_outliers_mask = (z_scores < threshold).all(axis=1)
#     df_no_outliers = df[no_outliers_mask]
#     target_no_outliers = target[no_outliers_mask]
#     return df_no_outliers, target_no_outliers

# # Remove outliers from your data
# X_train, y_train = remove_outliers(X_train_scaled, y_train)

# Build the XGBoost model using the filtered data
model_xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
model_xgb.fit(X_train, y_train)

# Make predictions on the test set
# X_test, y_test = remove_outliers(X_test_scaled, y_test)
predictions_xgb = model_xgb.predict(X_test)

# Calculate MSE for the filtered data
mse_xgb = mean_absolute_error(y_test, predictions_xgb)
print(f'Mean Absolute Error on Test Data (XGBoost - No Outliers): {mse_xgb:.2f}')

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # 3D projection

# Scatter plot of Biomipedance Index, Total Body Water, and Height
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], y_test, label='Actual', color='blue', alpha=0.6)
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], predictions_xgb, label='Predicted', color='red', alpha=0.6)

# Set labels for each axis
ax.set_xlabel('Biomipedance Index')
ax.set_ylabel('Height (cm)')
ax.set_zlabel('Total Body Water')

# Add a legend with MSE
ax.legend(labels=[f'MSE: {mse_xgb:.2f}'], loc='upper right')

plt.title("Total Body Water vs. Biomipedance Index and Height : XGB")
plt.show()


model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)
mse_rf = mean_absolute_error(y_test, predictions_rf)

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # 3D projection

# Scatter plot of Biomipedance Index, Total Body Water, and Height
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], y_test, label='Actual', color='blue', alpha=0.6)
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], predictions_rf, label='Predicted', color='red', alpha=0.6)

# Set labels for each axis
ax.set_xlabel('Biomipedance Index')
ax.set_ylabel('Height (cm)')
ax.set_zlabel('Total Body Water')

# Add a legend with MSE
ax.legend(labels=[f'MSE: {mse_rf:.2f}'], loc='upper right')

plt.title("Total Body Water vs. Biomipedance Index and Height : RF")
plt.show()

from sklearn.svm import SVR

model_svr = SVR(kernel='rbf')
model_svr.fit(X_train, y_train)
predictions_svr = model_svr.predict(X_test)
mse_svr = mean_absolute_error(y_test, predictions_svr)

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # 3D projection

# Scatter plot of Biomipedance Index, Total Body Water, and Height
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], y_test, label='Actual', color='blue', alpha=0.6)
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], predictions_svr, label='Predicted', color='red', alpha=0.6)

# Set labels for each axis
ax.set_xlabel('Biomipedance Index')
ax.set_ylabel('Height (cm)')
ax.set_zlabel('Total Body Water')

# Add a legend with MSE
ax.legend(labels=[f'MSE: {mse_svr:.2f}'], loc='upper right')

plt.title("Total Body Water vs. Biomipedance Index and Height : SVR")
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
ax.legend(labels=[f'MSE: {mse_nn:.2f}'], loc='upper right')

plt.title("Total Body Water vs. Biomipedance Index and Height : NN")
plt.show()

from sklearn.ensemble import GradientBoostingRegressor

model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_gb.fit(X_train, y_train)
predictions_gb = model_gb.predict(X_test)
mse_gb = mean_absolute_error(y_test, predictions_gb)

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')  # 3D projection

# Scatter plot of Biomipedance Index, Total Body Water, and Height
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], y_test, label='Actual', color='blue', alpha=0.6)
ax.scatter(X_test['Biomipedance Index'], X_test['height_cm'], predictions_gb, label='Predicted', color='red', alpha=0.6)

# Set labels for each axis
ax.set_xlabel('Biomipedance Index')
ax.set_ylabel('Height (cm)')
ax.set_zlabel('Total Body Water')

# Add a legend with MSE
ax.legend(labels=[f'MSE: {mse_gb:.2f}'], loc='upper right')

plt.title("Total Body Water vs. Biomipedance Index and Height : GB")
plt.show()