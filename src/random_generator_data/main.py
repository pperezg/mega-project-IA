import src.random_data_generator as rdg
import src.constants as cons
import numpy as np
import os

def verify_directory(path: str):
    """
    Verify if a directory at the specified path exists and create it if not.

    Parameters:
        path (str): The path to the directory.

    Returns:
        None
    """
    if not os.path.exists(path):
        os.mkdir(path)

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
    parameter_keys = list(cons.parameters.keys())

    # Generate data and save to files for each set of parameters
    for parameter_key in parameter_keys[2:]:
        matrix = rdg.generate_random_points(
            cons.parameters[parameter_keys[0]],
            cons.parameters[parameter_keys[1]],
            cons.parameters[parameter_key]["mu"],
            cons.parameters[parameter_key]["sigma"],
            cons.parameters[parameter_key]["q"]
        )
        np.savetxt(
            os.path.join(points_directory,
                         f'mu{cons.parameters[parameter_key]["mu"]}_sigma{cons.parameters[parameter_key]["sigma"]}_q{cons.parameters[parameter_key]["q"]}.csv'),
            matrix,
            delimiter=",")

        df = rdg.calculate_mu_sigma(matrix)
        df.to_csv(
            os.path.join(mu_sigma_directory,
                         f'mu{cons.parameters[parameter_key]["mu"]}_sigma{cons.parameters[parameter_key]["sigma"]}_q{cons.parameters[parameter_key]["q"]}.csv'),
            index=False)