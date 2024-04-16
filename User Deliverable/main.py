import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import signal
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

def calculate_baseline(df):
    weights = norm.pdf([1,2,3,4,5,6], 6, 2)
    weights /= weights.sum()
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

def calculate_hydrated_state(df, baseline, dehydration_deviation):
    deviations = df['TBW Equation'] - baseline
    hydration_scores = []
    hydration_categories = []
    print(4*dehydration_deviation)
    for deviation in deviations:
        print(deviation)
        
        if deviation > -4*dehydration_deviation:
            hydration_category = 'Hydrated'
            hydration_score = 1 + (deviation / (16*dehydration_deviation))
        elif deviation > -8*dehydration_deviation:
            hydration_category = 'Moderately Dehydrated'
            hydration_score = max(0, min(1, 1 + (deviation / (16*dehydration_deviation))))
        else:
            hydration_category = 'Severely Dehydrated'
            hydration_score = max(0, min(1, 1 + (deviation / (16*dehydration_deviation))))
        
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
        baseline = calculate_baseline(df)
        print(baseline)
        baseline_deviation = baseline - df['TBW Equation'].iloc[0]
        print(baseline_deviation)
    else:
        print("No baseline file found.")

    height, weight, age, gender = get_user_information()

    new_measurement_path = os.path.join(base_directory, 'new measurements')
    new_measurements = pd.DataFrame(columns=['R', 'Xc'])
    for measurement_file in os.listdir(new_measurement_path):
        date = measurement_file[9:13] + '-' + measurement_file[13:15] + '-' + measurement_file[15:17] + ' ' + measurement_file[18:20] + ':' + measurement_file[20:22] + ':' + measurement_file[22:24]
        date = pd.to_datetime(date)
        file_path = os.path.join(new_measurement_path, measurement_file)
        I, Q = read_data_from_file(file_path)
        R, Xc = data_preprocessing(I, Q)

        temp_df = pd.DataFrame({'Date': [date], 'R': [R], 'Xc': [Xc]})
        new_measurements = pd.concat([new_measurements, temp_df])
    
    new_measurements.set_index('Date', inplace=True)
    new_measurements.sort_index(inplace=True)
    
    new_measurements['TBW Equation'] = ((1.2 + 0.45 * height**2 / (new_measurements['R'] * 0.48 + 187.56) + 0.18 * weight) - 40.58) * (0.30/0.50) + 40.58
    
    new_measurements['hydration category'], new_measurements['hydration score'] = calculate_hydrated_state(new_measurements,baseline,baseline_deviation)
    print(new_measurements)

    
    color_map = {'Hydrated': 'green', 'Moderately Dehydrated': 'orange', 'Severely Dehydrated': 'red'}

    # Plot hydration Index with color mapping
    for category, score, date in zip(new_measurements['hydration category'], new_measurements['hydration score'], new_measurements.index):
        plt.scatter(date, score, color=color_map[category], marker='o')
    plt.plot(new_measurements.index,[0.75]*len(new_measurements.index),color='orange')
    plt.plot(new_measurements.index,[0.5]*len(new_measurements.index),color='red')
    plt.gcf().autofmt_xdate()
    plt.ylabel('Hydration Score')
    plt.xlabel('Time')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()