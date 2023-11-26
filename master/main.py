from dataReading import *
from generator import *
from SVMfisher import *
from SVMbasic import *
import numpy as np
import os
import sys
import itertools
from tqdm import tqdm
import pandas as pd

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
    data_space_directory = os.path.join(data_directory, "data_space")
    verify_directory(data_space_directory)

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

    ##################################################### Prepare to run all SVMs

    ts = [0.1,0.5,0.6, 0.7, 0.8, 0.9, 1]
    qs = [1, 1.2, 1.5, 1.7, 2, 2.5, 2.8]

    space_ix=itertools.combinations(combinations, r=2)
    kernel_ix=[('rbf',''),('lineal',''),('poli','')]+[('fisher','t{:.1f}_q{:.1f}'.format(t,q)) for t,q in itertools.product(ts,qs)]
    svm_results=pd.DataFrame(columns=['accuracy'],
                            index=pd.MultiIndex.from_tuples([tuple(itertools.chain(*row)) for row in itertools.product(space_ix,kernel_ix)],
                                                            names=['(mu0,sigma0,q0)','(mu1,sigma1,q1)','SVM kernel','kernel parameters']))
    
    ##################################################### Work by Pedro's team

    for theta_0, theta_1 in tqdm(svm_results.index.droplevel(['SVM kernel','kernel parameters']).drop_duplicates()):
        # Load data
        df_ij=pd.read_csv(os.path.join(data_space_directory,'mu%s_sigma%s_q%s_mu%s_sigma%s_q%s.csv'%(*theta_0,*theta_1)))
        x=df_ij[['mu','sigma']].to_numpy()
        y=df_ij[['label']].to_numpy().squeeze()

        # Train SVMs
        svms = train_svms(x,y)
        kernels_accuracy=evaluate_kernels(svms, x, y)
        svm_results.loc[(theta_0,theta_1,['rbf','lineal','poli']),:]=list(kernels_accuracy)

    svm_results.loc[(slice(None),slice(None),['rbf','lineal','poli']),:].to_csv(os.path.join(data_directory,'svm_results_a.csv'))

    print('Terminó trabajo de Pedro')
    
############################ Work by Sofi & Abe's team

    for theta_0, theta_1 in svm_results.index.droplevel(['SVM kernel','kernel parameters']).drop_duplicates():
        # Load data
        df_ij=pd.read_csv(os.path.join(data_space_directory,'mu%s_sigma%s_q%s_mu%s_sigma%s_q%s.csv'%(*theta_0,*theta_1)))
        x=df_ij[['mu','sigma']].to_numpy()
        y=df_ij[['label']].to_numpy().squeeze()

        f_acc=run_all_configurations_from_space(x, y, ts, qs)
        svm_results.loc[(theta_0, theta_1,'fisher', 't'+f_acc['t'].astype(str)+'_q'+f_acc['q'].astype(str)),:]=f_acc['fisher'].to_numpy()

    svm_results.to_csv(os.path.join(data_directory,'svm_results_b.csv'))

    print('Terminó trabajo de Sofi y Abe')