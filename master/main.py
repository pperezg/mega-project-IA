from dataReading import *
from generator import *
from SVM_basic import *
from SVM_fisher import *

import numpy as np
import os
import sys

###################################### Main de Paty
if __name__ == "__main__":
    # Set up directory paths
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, "data")
    verify_directory(data_directory)
    points_directory = os.path.join(data_directory, "points")
    verify_directory(points_directory)
    mu_sigma_directory = os.path.join(data_directory, "mu_sigma")
    verify_directory(mu_sigma_directory)

    # Get parameter keys
    parameter_keys = list(parameters.keys())

    # Generate data and save to files for each set of parameters
    for parameter_key in parameter_keys[2:]:
        matrix = generate_random_points(
            parameters[parameter_keys[0]],
            parameters[parameter_keys[1]],
            parameters[parameter_key]["mu"],
            parameters[parameter_key]["sigma"],
            parameters[parameter_key]["q"]
        )
        np.savetxt(
            os.path.join(points_directory,
                         f'mu{parameters[parameter_key]["mu"]}_sigma{parameters[parameter_key]["sigma"]}_q{parameters[parameter_key]["q"]}.csv'),
            matrix,
            delimiter=",")

        df = calculate_mu_sigma(matrix)
        df.to_csv(
            os.path.join(mu_sigma_directory,
                         f'mu{parameters[parameter_key]["mu"]}_sigma{parameters[parameter_key]["sigma"]}_q{parameters[parameter_key]["q"]}.csv'),
            index=False)

    # Plot the points and save them to a png file
    create_data_space((30, 6, 1), (20, 5, 1))

################## Main de Juli y Bombi
'''

def main():
    # Random example
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    # Use an additional parameter called heat_kernel with your kerenl function
    # or matrix.
    svms = train_svms(X, y)
    evaluate_kernels(svms, X, y)


if __name__ == "__main__":
    main()
'''

############################ Main de Sofi y Abe
'''
def main(ts, qs, experiments, graph):

    for name_exp in experiments:
        path = f'data/{name_exp}'
        X, labels = read_data(path)

        create_folders_if_not_exist([f'results/{name_exp}'])
        run_all_configurations(X, labels, ts, qs, name_exp, f'results/results_fisher_{name_exp}.csv', graph=graph)

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('Asumiendo argumentos por derecho')
        experiments = ['exp1', 'exp2', 'exp3', 'exp4']
        ts = [0.1,0.5,0.6, 0.7, 0.8, 0.9, 1]
        qs = [1, 1.2, 1.5, 1.7, 2, 2.5, 2.8]
        graph = False
    elif len(sys.argv) == 5:
        print('Leyendo argumentos')
        experiments_str = sys.argv[1]
        ts_str = sys.argv[2]
        qs_str = sys.argv[3]
        graph = sys.argv[4] == 'True'

        experiments = experiments_str.split(',')
        ts = [float(t) for t in ts_str.split(',')]
        qs = [float(q) for q in qs_str.split(',')]
        
    else:
        print('Error en los argumentos. Deben ser: [experiments] [ts] [qs] [graph]')
        sys.exit()

    main(ts, qs, experiments,graph)
'''