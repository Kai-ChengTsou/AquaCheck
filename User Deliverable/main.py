import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import signal
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from joblib import load

def calculate_baseline(df, model = True):
    weights = norm.pdf([1,2,3,4,5,6], 6, 2)
    weights /= weights.sum()
    if model:
        baseline = np.dot(weights, df['TBW Model'])
    else:
        baseline = np.dot(weights, df['TBW Equation'])
    return baseline

def read_data_from_file(filename):
    df = pd.read_csv(filename, on_bad_lines='skip')[10:-10]
    d_I = df.iloc[:,2].values
    d_Q = df.iloc[:,3].values
    data_I = d_I.astype(float) * 0.00662
    data_Q = d_Q.astype(float) * 0.00662

    return data_I, data_Q

def data_preprocessing(I, Q):
    b, a = signal.butter(3, 0.02)
    I = signal.filtfilt(b, a, I)
    Q = signal.filtfilt(b, a, Q)

    kdeI = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(I.reshape(-1,1))
    densityI = kdeI.score_samples(I.reshape(-1,1))

    kdeQ = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(Q.reshape(-1,1))
    densityQ = kdeQ.score_samples(Q.reshape(-1,1))

    R = I[max(enumerate(densityI),key=lambda x: x[1])[0]]
    Xc = Q[max(enumerate(densityQ),key=lambda x: x[1])[0]]

    return R, Xc

def get_user_information():
    height = float(input("Enter your height in centimeters: "))
    weight = float(input("Enter your weight in kilograms: "))
    age = int(input("Enter your age in years: "))
    gender = int(input("Enter your gender (1-Male/2-Female): "))
    return height, weight, age, gender

def calculate_hydrated_state(df, baseline, dehydration_deviation, model=True):
    if model:
        deviations = df['TBW Model'] - baseline
    else:
        deviations = df['TBW Equation'] - baseline
    hydration_scores = []
    hydration_categories = []
    print(f'Dehydration threshold:        {5*dehydration_deviation:.2f} kg (or L)')
    print(f'Severe dehydration threshold: {10*dehydration_deviation:.2f} kg (or L)')
    for deviation in deviations:
        
        if deviation > -5*dehydration_deviation:
            hydration_category = 'Hydrated'
            hydration_score = 1 + (deviation / (20*dehydration_deviation))
        elif deviation > -10*dehydration_deviation:
            hydration_category = 'Moderately Dehydrated'
            hydration_score = max(0, min(1, 1 + (deviation / (20*dehydration_deviation))))
        else:
            hydration_category = 'Severely Dehydrated'
            hydration_score = max(0, min(1, 1 + (deviation / (20*dehydration_deviation))))
        
        # Append hydration score and category to the lists
        hydration_categories.append(hydration_category)
        hydration_scores.append(hydration_score)
    
    return hydration_categories, hydration_scores

def main():
    base_directory = os.path.dirname(os.path.abspath(__file__))
                                     
    historical_path = os.path.join(base_directory, 'historical', 'baseline.csv')
    if os.path.exists(historical_path):
        df = pd.read_csv(historical_path,index_col='Date')
        #print(f"Baseline data:\n{df}")
        baseline_model = calculate_baseline(df, True)
        baseline_equation = calculate_baseline(df, False)
        print(f'Model baseline:    {baseline_model:.2f}')
        print(f'Equation baseline: {baseline_equation:.2f}')
        baseline_deviation_model = baseline_model - df['TBW Model'].iloc[0]
        baseline_deviation_equation = baseline_equation - df['TBW Equation'].iloc[0]
        print(f'Baseline model deviation:    {baseline_deviation_model:.2f}')
        print(f'Baseline equation deviation: {baseline_deviation_equation:.2f}')
    else:
        print("No baseline file found.")

    new_measurement_path = os.path.join(base_directory, 'new measurements')
    new_measurements = pd.DataFrame(columns=['R', 'Xc'])
    height, weight, age, gender = get_user_information()
    for measurement_file in os.listdir(new_measurement_path):
        date = measurement_file[9:13] + '-' + measurement_file[13:15] + '-' + measurement_file[15:17] + ' ' + measurement_file[18:20] + ':' + measurement_file[20:22] + ':' + measurement_file[22:24]
        date = pd.to_datetime(date)
        file_path = os.path.join(new_measurement_path, measurement_file)
        I, Q = read_data_from_file(file_path)
        R, Xc = data_preprocessing(I, Q)

        temp_df = pd.DataFrame({'Date': [date], 'R': [R], 'Xc': [Xc]})
        new_measurements = pd.concat([new_measurements, temp_df])
    new_measurements['Height'] = height
    new_measurements['Weight'] = weight
    new_measurements['Age'] = age
    new_measurements['Gender'] = gender
    new_measurements.set_index('Date', inplace=True)
    new_measurements.sort_index(inplace=True)
    new_measurements['Height to Weight'] = height/weight
    new_measurements['BMI'] = weight / (height / 100) ** 2
    new_measurements['Adjusted R'] = new_measurements['R'] * 0.48 + 187.56
    new_measurements['BSA'] = 0.007184 * height ** 0.725 * weight ** 0.425
    new_measurements['BioImpedance Index'] = (height ** 3 / new_measurements['Adjusted R'])
    new_measurements['TBW Equation'] = ((1.2 + 0.45 * height**2 / (new_measurements['Adjusted R']) + 0.18 * weight) - 40.58) * (0.30/0.50) + 40.58
    
    stacking_model_tbw = load('model/stacking_model_tbw.joblib')
    x_dehydrated = new_measurements[['Gender', 'Age', 'Weight', 'Height', 'BMI', 'BSA', 'Height to Weight', 'BioImpedance Index']]
    
    new_measurements['TBW Model'] = stacking_model_tbw.predict(x_dehydrated)
    new_measurements['hydration category model'], new_measurements['hydration score model'] = calculate_hydrated_state(new_measurements,baseline_model,baseline_deviation_model)
    new_measurements['hydration category equation'], new_measurements['hydration score equation'] = calculate_hydrated_state(new_measurements,baseline_equation,baseline_deviation_equation,False)


    print(new_measurements)

    color_map_model = {'Hydrated': 'green', 'Moderately Dehydrated': 'orange'} # 'Severely Dehydrated': 'red'
    color_map_equation = {'Hydrated': '#76b947', 'Moderately Dehydrated': '#cc8400'}
    fig, ax = plt.subplots()
    # Plot hydration Index with color mapping
    for category, score, date in zip(new_measurements['hydration category model'], new_measurements['hydration score model'], new_measurements.index):
        ax.scatter(date, score, color=color_map_model[category], marker='o')
    for i, category in enumerate(color_map_model):
        ax.scatter([], [], color=color_map_model[category], label=category + ' model')  

    for category, score, date in zip(new_measurements['hydration category equation'], new_measurements['hydration score equation'], new_measurements.index):
        ax.scatter(date, score, color=color_map_equation[category], marker='o')
    for i, category in enumerate(color_map_equation):
        ax.scatter([], [], color=color_map_equation[category], label=category + ' equation') 
 
    ax.plot(new_measurements.index,[0.75]*len(new_measurements.index),color='orange')
    ax.plot(new_measurements.index,[0.5]*len(new_measurements.index),color='red')
    # ax.gcf().autofmt_xdate()
    ax.set_ylabel('TBW (kg)')
    ax.set_xlabel('Time')
    ax.set_xticklabels(['11AM','12PM','1PM','2PM','3PM','4PM'])
    ax.set_title('Hydration Score on Dehydrated Bar Test')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()