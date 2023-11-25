from dataReading import *
from generator import *
from SVMfisher import *
from SVMbasic import *
import numpy as np
import os
import sys
import itertools
from tqdm import tqdm

if __name__ == "__main__":
    ##################################################### Work by Patiño's team
    # Set up directory paths
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, "GeneratedData")
    verify_directory(data_directory)
    points_directory = os.path.join(data_directory, "points")
    verify_directory(points_directory)
    mu_sigma_directory = os.path.join(data_directory, "mu_sigma")
    verify_directory(mu_sigma_directory)

    mu = [0,5,10]
    sigma = [1,3,5]
    q = [1,1.5,2]
    parameters = {"num_points": 100,"dimension": 50}
    combinations = list(itertools.product(mu, sigma, q))

    # Get parameter keys
    parameter_keys = list(parameters.keys())

    # Generate data and save to files for each set of parameters
    for combination in combinations:
        matrix = generate_random_points(
            parameters[parameter_keys[0]],
            parameters[parameter_keys[1]],
            combination[0],
            combination[1],
            combination[2]
        )
        np.savetxt(
            os.path.join(points_directory,
                         f'mu{combination[0]}_sigma{combination[1]}_q{combination[2]}.csv'),
            matrix,
            delimiter=",")

        df = calculate_mu_sigma(matrix)
        df.to_csv(
            os.path.join(mu_sigma_directory,
                         f'mu{combination[0]}_sigma{combination[1]}_q{combination[2]}.csv'),
            index=False)

    print('Terminó trabajo de Patiño')

    ##################################################### Work by Agustin's team
    
    for i in tqdm(range(len(combinations))):
        for j in range(len(combinations)):
            if i != j:
                create_data_space(combinations[i], combinations[j])

    print('Terminó trabajo de Agustin')

    ##################################################### Work by Pedro's team
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