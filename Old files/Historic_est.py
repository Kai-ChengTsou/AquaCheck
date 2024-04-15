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
import os

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

def convert_to_time_from_start(day, hour):
    return_hours = 0
    morning_afternoon = hour[-2:]
    time = hour[:-2]
    if day == 'Sunday':
        return_hours += 24
    elif day == 'Monday':
        return_hours += 24 * 2
    elif day == 'Tuesday':
        return_hours += 24 * 3
    elif day == 'Wednesday':
        return_hours += 24 * 4
    elif day == 'Thursday':
        return_hours += 24 * 5
    elif day == 'Friday':
        return_hours += 24 * 6

    if morning_afternoon == 'PM':
        return_hours += 12

    return return_hours + int(time)

def run_model():
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

  directory_path = './MLTesting/Datasets/Week Long Test/'

  # Initialize lists to store the processed data
  processed_data = []
  height = 173
  weight = 69
  age = 22

  # Iterate over each day and time
  for root, dirs, files in os.walk(directory_path):
      for file in files:
          if file.endswith('50kHz.csv'):
              # Extract day and time from the directory path
              day, time = os.path.split(root)[-2:]
              day = day.split('/')[-1]

              # Construct the file path
              file_path = os.path.join(root, file)

              # Read the data from the file
              data = pd.read_csv(file_path)
              data = data[5:-4]

              # Process the data
              data_I = (data.iloc[:, 2].values.astype(float) * 0.00662)
              data_Q = (data.iloc[:, 3].values.astype(float) * 0.00662)

              nansI = np.isnan(data_I)
              nansQ = np.isnan(data_Q)

              I_func_pchip = PchipInterpolator(np.arange(len(data_I[~nansI])), data_I[~nansI])
              Q_func_pchip = PchipInterpolator(np.arange(len(data_Q[~nansQ])), data_Q[~nansQ])

              b, a = signal.butter(3, 0.02)
              I = signal.filtfilt(b, a, data_I)
              Q = signal.filtfilt(b, a, data_Q)

              kdeI = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(I.reshape(-1, 1))
              densityI = kdeI.score_samples(I.reshape(-1, 1))

              kdeQ = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(Q.reshape(-1, 1))
              densityQ = kdeQ.score_samples(Q.reshape(-1, 1))

              R = I[max(enumerate(densityI), key=lambda x: x[1])[0]] * 0.48 + 187.56
              Xc = Q[max(enumerate(densityQ), key=lambda x: x[1])[0]] 

              Z = np.sqrt(np.square(np.average(I)) + np.square(np.average(Q))) * 0.48 + 187.56
              Pha = np.arctan2(np.average(Q), np.average(I))

              # Print the processed results
              print(f'\nDay: {day}, Time: {time}')
              print(f'R: {R:.2f} Ohms')
              print(f'Xc: {Xc:.2f} Ohms')
              print(f'Z: {Z:.2f} Ohms')
              print(f'Pha: {Pha:.2f} Radians')

              # Store the processed data
              processed_data.append({
                  'Day': day,
                  'Time': time,
                  'R': R,
                  'Xc': Xc,
                  'Z': Z,
                  'Pha': Pha,
                  'BioImpedance Index': (height ** 3 / R),
                  'Height': height,
                  'Weight': weight,
                  'BMI': weight / (height / 100) ** 2,
                  'BSA': 0.007184 * height ** 0.725 * weight ** 0.425,
                  'Height to Weight': height / weight,
                  'Gender': 1,
                  'Age': age
              })

  # Convert the processed data to a DataFrame
  processed_df = pd.DataFrame(processed_data)
  processed_df = processed_df.iloc[1:] #Neglect the first entry which is invalid

  # Display or further process the DataFrame as needed

  print(f'\n{processed_df}\n')

  # Plot the processed data in four separate subplots
  fig, axes = plt.subplots(2, 2, figsize=(12, 6))

  # Plot R
  axes[0, 0].scatter(processed_df['Time'], processed_df['R'], marker='^', label='R', color='b')
  axes[0, 0].set_title('R')
  axes[0, 0].set_xlabel('Time')
  axes[0, 0].set_ylabel('Ohms')
  axes[0, 0].tick_params(axis='x', rotation=45)
  axes[0, 0].grid(True, linestyle='--', alpha=0.5)

  # Plot Xc
  axes[0, 1].scatter(processed_df['Time'], processed_df['Xc'], marker='*', label='Xc', color='r')
  axes[0, 1].set_title('Xc')
  axes[0, 1].set_xlabel('Time')
  axes[0, 1].set_ylabel('Ohms')
  axes[0, 1].tick_params(axis='x', rotation=45)
  axes[0, 1].grid(True, linestyle='--', alpha=0.5)

  # Plot Z
  axes[1, 0].scatter(processed_df['Time'], processed_df['Z'], marker='x', label='Z', color='m')
  axes[1, 0].set_title('Z')
  axes[1, 0].set_xlabel('Time')
  axes[1, 0].set_ylabel('Ohms')
  axes[1, 0].tick_params(axis='x', rotation=45)
  axes[1, 0].grid(True, linestyle='--', alpha=0.5)

  # Plot Pha
  axes[1, 1].scatter(processed_df['Time'], processed_df['Pha'], marker='o', label='Pha', color='c')
  axes[1, 1].set_title('Pha')
  axes[1, 1].set_xlabel('Time')
  axes[1, 1].set_ylabel('Radians')
  axes[1, 1].tick_params(axis='x', rotation=45)
  axes[1, 1].grid(True, linestyle='--', alpha=0.5)

  plt.tight_layout()
  plt.show()

  # Plot the processed data in one plot
  plt.figure(figsize=(12, 5))

  # Plot R after processing
  plt.scatter(processed_df['Time'], processed_df['R'], marker='^', label='R after processing', color='b')
  plt.axhline(y = np.average(processed_df['R']), color='gray', linestyle='--', label=f'R after processing Average: {np.average(processed_df["R"]):.2f} Ohms')

  # Plot R before processing
  plt.scatter(processed_df['Time'], (processed_df['R'] - 187.56) / 0.48, marker='x', label='R before processing', color='m')
  plt.axhline(y = np.average(processed_df['R'] * 1.33), color='black', linestyle='--', label=f'R before processing Average: {np.average(processed_df["R"] * 1.33):.2f} Ohms')

  # Set plot title and labels
  plt.title('Processed Data Comparison')
  plt.xlabel('Time')
  plt.ylabel('Ohms')

  # # Rotate x-axis labels
  # plt.xticks(rotation=45)

  # Add gridlines
  plt.grid(True, linestyle='--', alpha=0.5)

  # Add legend
  plt.legend()

  # Show plot
  plt.tight_layout()
  plt.show()
    
  # Prepare the features for prediction
  x_dehydrated = processed_df[['Gender', 'Age', 'Weight', 'Height', 'BMI', 'BSA', 'Height to Weight', 'BioImpedance Index']]

  # Make predictions using stacking model
  tbw_pred_stack = stacking_model_tbw.predict(x_dehydrated)

  # Add the predictions to the processed DataFrame
  processed_df['TBW Prediction'] = tbw_pred_stack  # Adjust as needed

  # Plot the TBW predictions
  plt.figure(figsize=(12, 6))

  # Plot TBW predictions
  plt.scatter(processed_df['Day'], processed_df['TBW Prediction'], marker='x', color='b', label='TBW Prediction')

  # Set plot title and labels
  plt.title('TBW Predictions')
  plt.xlabel('Day')
  plt.ylabel('Prediction')
  plt.xticks(rotation=45)
  plt.grid(True, linestyle='--', alpha=0.5)

  # Show plot
  plt.tight_layout()
  plt.show()

  # Print the day and estimated TBW
  for index, row in processed_df.iterrows():
    print(f"Day: {row['Day']}, Estimated TBW: {row['TBW Prediction']:.2f}")
    
  tests = ['8AM', '10AM', '12PM', '1230PM', '1PM', '2PM', '4PM', '6PM']
  data = []
  data_I = []
  data_Q = []

  for time in tests:
    test_data = pd.read_csv('./MLTesting/Datasets/' + time + ' Standard.csv')[10:-4]
    data.append(test_data)
    temp_data_I = test_data.iloc[:,2].values
    temp_data_Q = test_data.iloc[:,3].values

    # data_I.append(temp_data_I.astype(float) * 0.016 + 14)
    # data_Q.append(temp_data_Q.astype(float) * 0.016 + 14)

  # R_post = []
  # Xc_post = []

  for i in range(len(data_I)):
    b, a = signal.butter(3, 0.02)
    data_I[i] = signal.filtfilt(b, a, data_I[i])
    data_Q[i] = signal.filtfilt(b, a, data_Q[i])

  for i in range(len(data_I)):
    kdeI = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_I[i].reshape(-1,1))
    densityI = kdeI.score_samples(data_I[i].reshape(-1,1))

    kdeQ = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data_Q[i].reshape(-1,1))
    densityQ = kdeQ.score_samples(data_Q[i].reshape(-1,1))

    # R_post.append(data_I[i][max(enumerate(densityI),key=lambda x: x[1])[0]])
    # Xc_post.append(data_Q[i][max(enumerate(densityQ),key=lambda x: x[1])[0]])

  # Imp_post = (np.square(Xc_post) + np.square(R_post))**(1/2)
  # Pha_post = np.rad2deg(np.arctan2(Xc_post, R_post))
    
  R_post = [669.9105768313373, 680.625465485632, 688.3550312609393, 703.5778493611149, 688.2165668692417, 669.277104945529, 652.19874692870655]
  Xc_post = [-59.16480156201287, -65.89451013525705, -86.08025034482249, -88.13972781556944, -85.68358641838101, -82.22261865511265, -79.42146879378987]
  Imp_post = [672.5181445094008, 683.8078024807108, 693.7164107628391, 709.0771479402042, 693.5298983423869, 674.308832971045]
  Pha_post = [-5.0471225088293385, -5.529836111974436, -7.127956182507452, -7.140450602177827, -7.096858606040883, -7.003856125239041]

  Imp_post.append((np.square(Xc_post[-1]) + np.square(R_post[-1]))**(1/2))
  Pha_post.append(np.rad2deg(np.arctan2(Xc_post[-1], R_post[-1])))

  R_post = np.array(R_post) * 0.48 + 187.56
  # weight_at_test = [67.5,67.7,67.4,68,68.7,68.8,69.2,68.9]
  # tbw_percent_at_test = [63.2,62.4,61.8,61.9,61.5,61.4,61.3,62]
  tbw_percent_at_test = [0.626,0.625,0.623,0.621,0.618,0.619,0.62]
  weight_at_test = [67.9,67.8,67.9,67.8,69.3,69.4,69.4]
  tbw_kg_at_test = []

  tbw_from_equation = []
  for i in range(len(weight_at_test)):
    tbw_kg_at_test.append(weight_at_test[i]*tbw_percent_at_test[i])
    tbw_from_equation.append(1.2 + 0.45 * 173**2 / R_post[i] + 0.18 * 70)

  dehydrated_test_data = pd.DataFrame({
    'Gender': [1]*7, 
    'Weight': [70]*7,
    'Height': [173]*7, 
    'Age': [22.52]*7,
    'R': R_post})

  dehydrated_test_data['BioImpedance Index'] = (dehydrated_test_data['Height'] ** 3) / dehydrated_test_data['R']
  dehydrated_test_data['BMI'] = dehydrated_test_data['Weight'] / (dehydrated_test_data['Height'] / 100) ** 2
  dehydrated_test_data['BSA'] = 0.007184 * dehydrated_test_data['Height'] ** 0.725 * dehydrated_test_data['Weight'] ** 0.425
  dehydrated_test_data['Height to Weight'] = dehydrated_test_data['Height'] / dehydrated_test_data['Weight']

  x_tbw_dehydrated = dehydrated_test_data[['Gender','Age','Weight','Height','BMI','BSA','Height to Weight','BioImpedance Index']]

  tbw_pred_stack = stacking_model_tbw.predict(x_tbw_dehydrated)

  plt.figure(figsize=(12, 5))
  time_of_day = [11,12,13,14,15,16,17]
  # Plot for TBW
  plt.scatter(time_of_day, tbw_pred_stack, label='Model Prediction', marker='o', color='blue', alpha=0.5)
  plt.scatter(time_of_day, tbw_kg_at_test, label='Ground Truth', marker='x', color='black', alpha=0.5)
  plt.scatter(time_of_day, tbw_from_equation, label='Equation', marker='^', color='red', alpha=0.5)

  # Add grid
  plt.grid(True, linestyle='--', alpha=0.5)

  plt.title('Dehydrated Test - Estimated TBW (Using Gel Electrodes)')
  plt.ylabel('TBW (kg)')
  plt.xlabel('Time of day (h)')
  plt.legend()
  plt.show()

  # Get the average of TBW estimates
  tbw_predictions = np.average(processed_df['TBW Prediction'])

  # Calculate the average of the top 5 predictions or all available predictions if fewer than 5
  average_tbw = np.mean(tbw_predictions)
  print(f"\nAverage of TBW predictions: {average_tbw:.2f} Kg")
  print(f"Average of Actual TBW: {np.average(tbw_kg_at_test):.2f} Kg")
  print(f"TBW standard deviation: {np.std(processed_df['TBW Prediction']):.2f}")

  # Calculate the deviation from the baseline TBW prediction
  tbw_deviation_from_baseline = tbw_pred_stack - average_tbw
  tbw_std_deviation = np.std(processed_df['TBW Prediction'])

  print('\nDehydrated test values deviation from baseline')
  for deviation in tbw_deviation_from_baseline:
      print(f"Deviation: {deviation:.2f} kg")
  
  # Define thresholds for dehydration based on standard deviations
  dehydration_threshold = 1.5 * tbw_std_deviation
  severe_dehydration_threshold = 3 * tbw_std_deviation

  print(f"Thresholds: {dehydration_threshold:.2f} kg, {severe_dehydration_threshold:.2f} kg")

  # Initialize lists to store hydration scores, categories, and values
  hydration_scores = []
  hydration_categories = []

  # Iterate through the deviations and determine hydration scores, categories, and values
  for deviation in tbw_deviation_from_baseline:
      if deviation > -dehydration_threshold:
          hydration_category = 'Hydrated'
          hydration_score = 1 + (deviation / severe_dehydration_threshold)
      elif -severe_dehydration_threshold < deviation:
          hydration_category = 'Moderately Dehydrated'
          hydration_score = max(0, min(1, 1 + (deviation / severe_dehydration_threshold)))
      else:
          hydration_category = 'Severely Dehydrated'
          hydration_score = max(0, min(1, 1 + (deviation / severe_dehydration_threshold)))
      
      # Append hydration score and category to the lists
      hydration_categories.append(hydration_category)
      hydration_scores.append(hydration_score)

  # Print hydration categories, scores, predicted TBW, and actual TBW
  print("\nHydration Categories, Scores, Predicted TBW, and Actual TBW:")
  for i, (category, score, tbw_prediction, tbw_actual) in enumerate(zip(hydration_categories, hydration_scores, tbw_pred_stack, tbw_kg_at_test)):
      print(f"{category} (Score: {score:.2f}), Predicted TBW: {tbw_prediction:.2f}, Actual TBW: {tbw_actual:.2f}")
  
  # Create subplots
  fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  color_map = {'Hydrated': 'green', 'Moderately Dehydrated': 'orange', 'Severely Dehydrated': 'red'}

  # Plot hydration Index with color mapping
  for category, score, time in zip(hydration_categories, hydration_scores, time_of_day):
      axes[0].scatter(time, score, color=color_map[category], marker='o')

  axes[0].set_ylabel('Hydration Score')
  axes[0].set_xlabel('Time of Day')
  axes[0].grid(True, linestyle='--', alpha=0.5)

  # Add labels for dots
  for i, category in enumerate(color_map):
      axes[0].scatter([], [], color=color_map[category], label=category)  # Create empty scatter plot for label
  axes[0].legend(loc='upper right')  # Add legend to show labels

  # Add dehydration threshold and severe dehydration threshold lines
  axes[1].axhline(y = average_tbw - dehydration_threshold, color='gray', linestyle='--', label='Dehydration Threshold')
  axes[1].axhline(y = average_tbw - severe_dehydration_threshold, color='red', linestyle='--', label='Severe Dehydration Threshold')
  axes[1].scatter(time_of_day, tbw_pred_stack, color='blue', marker='x')
  axes[1].axhline(y=average_tbw, color='black', linestyle='--', label='Baseline TBW')
  axes[1].set_ylabel('TBW Prediction')
  axes[1].set_xlabel('Time of Day')
  axes[1].legend()
  axes[1].grid(True, linestyle='--', alpha=0.5)

  # Adjust layout
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  run_model()
