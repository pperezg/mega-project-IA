import pandas as pd
import os
import matplotlib.pyplot as plt


def create_data_frames():
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


def create_data_space(first_source, scnd_source):
    # Each source argument is a triple (mu, sigma, q)

    # Assert that the arguments are triples
    assert len(first_source) == 3 and len(scnd_source) == 3
    # Assert that the triples are different
    assert first_source != scnd_source
    # Assert that the first triple corresponds to an existing file
    first_source_file_name = ('data/mu' + str(first_source[0]) + '_sigma' + str(first_source[1]) + '_q' +
                              "{:.1f}".format(first_source[2]) + '.csv')
    assert os.path.isfile(first_source_file_name)
    # Assert that the second triple corresponds to an existing file
    scnd_source_file_name = ('data/mu' + str(scnd_source[0]) + '_sigma' + str(scnd_source[1]) + '_q' +
                             "{:.1f}".format(scnd_source[2]) + '.csv')
    assert os.path.isfile(scnd_source_file_name)

    # Create a folder named 'data_space' in the current directory
    if not os.path.exists('data_space'):
        os.makedirs('data_space')
    # Read the data from the first source file
    first_source_data_frame = pd.read_csv(first_source_file_name)
    # Append a column containing the label of the source to the data frame
    first_source_data_frame['label'] = 0
    # Read the data from the second source file
    scnd_source_data_frame = pd.read_csv(scnd_source_file_name)
    # Append a column containing the label of the source to the data frame
    scnd_source_data_frame['label'] = 1
    # Create the data_space data frame by concatenating the two data frames
    data_space = pd.concat([first_source_data_frame, scnd_source_data_frame])
    # Save the data_space data frame to a csv file
    data_space.to_csv('data_space/mu' + str(first_source[0]) + '_sigma' + str(first_source[1]) + '_q' +
                      "{:.1f}".format(first_source[2]) + '_mu' + str(scnd_source[0]) + '_sigma' + str(scnd_source[1]) +
                      '_q' + "{:.1f}".format(scnd_source[2]) + '.csv', index = False)

    # Plot the points of the first source
    plt.scatter(first_source_data_frame['mu'], first_source_data_frame['sigma'], c='red')
    # Plot the points of the second source
    plt.scatter(scnd_source_data_frame['mu'], scnd_source_data_frame['sigma'], c = 'blue')
    # Add axis labels
    plt.xlabel('mu')
    plt.ylabel('sigma')
    # Add legend
    plt.legend(['μ = ' + str(first_source[0]) + ', σ = ' + str(first_source[1]) + ', q = ' +
                "{:.1f}".format(first_source[2]),
                'μ = ' + str(scnd_source[0]) + ', σ = ' + str(scnd_source[1]) + ', q = ' +
                "{:.1f}".format(scnd_source[2])])
    # Save the plot to a png file
    plt.savefig('data_space/mu' + str(first_source[0]) + '_sigma' + str(first_source[1]) + '_q' +
                "{:.1f}".format(first_source[2]) + '_mu' + str(scnd_source[0]) + '_sigma' + str(scnd_source[1]) +
                '_q' + "{:.1f}".format(scnd_source[2]) + '.png')


create_data_space((30, 6, 1), (20, 5, 1))
