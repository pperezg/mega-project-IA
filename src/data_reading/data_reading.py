import pandas as pd
import os


def create_data_points():
    # Create a folder named 'data' in the current directory
    if not os.path.exists('data'):
        os.makedirs('data')

    data_path = '../random_generator_data/data/points/'
    for file_name in os.listdir(data_path):
        if file_name.endswith('.csv'):
            numbers_generation_data_frame = pd.read_csv(data_path + file_name)
            # Create new dataframe containing the mean and standard deviation of each of the observations
            mean_std_data_frame = pd.DataFrame(columns=['mu', 'sigma'])
            mean_std_data_frame['mu'] = numbers_generation_data_frame.mean(axis = 1)
            mean_std_data_frame['sigma'] = numbers_generation_data_frame.std(axis = 1)
            # Save the dataframe to a csv file with the same name as the original file in the 'data' folder
            mean_std_data_frame.to_csv('data/' + file_name, index = False)


create_data_points()
