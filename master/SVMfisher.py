# libreries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay

# Importa las bibliotecas necesarias
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import plotly.express as px
import scipy.integrate as spi
from tqdm import tqdm
import os
import sys

def normalize_min_max(matrix):
    """
        Método para normalizar los datos
        Recibe: 
            1. matrix: Matriz de datos original
        Entrega: 
            1. normalized_data: Matriz de datos normalizados
    """
    max_values = np.max(matrix, axis=0)
    min_values = np.min(matrix, axis=0)

    # Lleva los datos al hiperplano (0, 1) utilizando la normalización min-max
    normalized_data = (matrix - min_values) / (max_values - min_values)
    return normalized_data

class FisherKernel():
    def __init__(self,t, q=1) -> None:
        self.t = t
        self.q = q
        const = 1/(2**(5/2) + np.pi**(3/2) * self.t**(-3/2))
        self.expr = const * np.exp(-self.t/4)
        

    def distH(self,X,Y):
        """
        rho = dist_h = arcosh

        Args:
            X1: Matriz de datos de forma (n_samples_1, 2).
            X2: Matriz de datos de forma (n_samples_2, 2).

        Returns:
            Matriz de distancias h de forma (n_samples_1, n_samples_2).
        """

        rho = np.zeros((X.shape[0],Y.shape[0]))
        for idx_x, xi in enumerate(X):
            for idx_y, yi in enumerate(Y):
                rho[idx_x,idx_y] = np.arccosh(1+ np.sum((xi-yi)**2)/(2*xi[1]*yi[1]))

        return rho

    def to_integrate(self,s,rho):
        return ((s*np.exp( -(s**2)/(4*self.t)))/( np.sqrt(np.cosh(s)-np.cosh(rho))) )
    
    def solve_integral(self, rhos):
        filas_rho, columnas_rho = rhos.shape

        K = np.zeros((filas_rho,columnas_rho))

        b = np.inf

        for fila in tqdm(range(filas_rho)):
            for columna in range(columnas_rho):
                result, _ = spi.quad(self.to_integrate, rhos[fila,columna], b, args=(rhos[fila,columna],))
                K[fila,columna] = result
        
        return K
    

    def kernelFisher(self, X, Y):

        X[:,1] = np.sqrt(3-self.q)*X[:,1]
        self.t = self.q/(3-self.q)*self.t
        return self.Kg_H(X,Y)


    def Kg_H(self,X,Y):
        """
        Kernel fisher.

        Args:
            X1: Matriz de datos de forma (n_samples_1, n_features).
            X2: Matriz de datos de forma (n_samples_2, n_features).

        Returns:
            Matriz de kernel de forma (n_samples_1, n_samples_2).
        """
        rhos = self.distH(X,Y)
        K = self.solve_integral(rhos)
        return self.expr * K
    

def plot_results(clf, X, y , kernel_name, params, exp):
    plt.figure()
    ax = plt.gca()

    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.Paired,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
    )

    title = f"2-Class classification using Support Vector Machine with {kernel_name} kernel"
    
    if params is not None:
        params_strs = ','.join([f'{name} = {params[name]}'for name in params])
        path = f"results/{exp}/{kernel_name}_{params_strs}.png"
    else:
        path = f"results/{exp}/{kernel_name}.png"

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")
    plt.title(title)
    plt.axis("tight")
    plt.savefig(path)

def create_folders_if_not_exist(folder_paths):
    """Function that creates folders if they don't exist
    Parameters
    ----------
    folder_paths : list
        List with the paths of the folders to create
    """
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)




def sample_data(X, label, RANDOM_SEED, val = False):

    if val:
    
        X_train, X_tv, y_train, y_tv = train_test_split(X, label ,test_size=0.4, random_state=RANDOM_SEED)
        X_test, X_val, y_test, y_val = train_test_split(X_tv, y_tv, test_size=0.5, random_state=RANDOM_SEED)

        return X_train, X_test, X_val, y_train, y_test, y_val
    
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, label ,test_size=0.2, random_state=RANDOM_SEED)

        return X_train, X_test, y_train, y_test


def sample_data_transform(X, label):
    _, indices_V, indices_T, indices_t = partitioning(X)
    X_T = transform_space(X)

    X_train = X[indices_T,:] # Datos originales train
    X_test = X[indices_t,:] # Datos originales test
    X_val = X[indices_V,:] # Datos originales validación

    y_train = label[indices_T] # Etiquetas train
    y_test = label[indices_t] # Etiquetas test
    y_val = label[indices_V] # Etiquetas validación

    X_T_train = X_T[indices_T,:] # Datos transformados train
    X_T_test = X_T[indices_t,:] # Datos transformados test
    X_T_val = X_T[indices_V,:] # Datos transformados validación

    return X_train, X_test, X_val, y_train, y_test, y_val, X_T_train, X_T_test, X_T_val


def partitioning(datos,size_S = 0.8,size_T= 0.6):
    """
        Método que hace el partitioning de los datos en el set de S, training, testing y validation
        Recibe:
            1. datos: Datos al cual se le va a hacer el partitioning
            2. Size_S: Tamaño del set S
            3. Size_T: Tamaño del set T (training)
        Entrega:
            1. S: Set S
            2. V: set de validación V
            3. T: set de training T
            4. t: set de testing t
    """
    indices_datos = np.arange(len(datos))
 
    indices_S = np.random.choice(indices_datos, size=int(round(len(datos)*size_S)),replace=False)
    indices_V = np.setdiff1d(indices_datos, indices_S)
    indices_T = np.random.choice(indices_S, size=int(round(len(datos)*size_T)),replace=False)
    indices_t = np.setdiff1d(indices_S, indices_T)

 
    return indices_S, indices_V, indices_T, indices_t


def transform_space(X):
    X1 = np.mean(X, axis=1)
    X2 = np.std(X, axis=1)
    T = np.concatenate((X1.reshape(-1, 1), X2.reshape(-1, 1)), axis=1)
    return T

def run_fisher_kernel(X_train, y_train, X_test, y_test, t, q, RANDOM_SEED):

    
    instance = FisherKernel(t, q)
    svm_fisher = SVC(kernel=instance.kernelFisher, random_state = RANDOM_SEED)
    svm_fisher.fit(X_train, y_train)
    accuracy_fisher = svm_fisher.score(X_test, y_test)
    fisher_pred = svm_fisher.predict(X_test)
    print('fisher_pred',fisher_pred)
    print("Accuracy con Kernel fisher personalizado: t:",t, " q:", q, accuracy_fisher)

    return accuracy_fisher, svm_fisher


def run_lineal_kernel(X_train, y_train, X_test ,y_test):
    svm_lineal = SVC(kernel='linear')
    svm_lineal.fit(X_train, y_train)
    accuracy_lineal = svm_lineal.score(X_test, y_test)
    # lineal_pred = svm_lineal.predict(X_test)
    print("Accuracy con Kernel lineal:", accuracy_lineal)
    return accuracy_lineal, svm_lineal

def run_rbf_kernel(X_train, y_train,X_test ,y_test):
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    accuracy_rbf = svm_rbf.score(X_test, y_test)
    # rbf_pred = svm_rbf.predict(X_test)
    print("Accuracy con Kernel rbf:", accuracy_rbf)
    return accuracy_rbf, svm_rbf

def run_all_configurations(X, label, ts, qs, exp, path, graph ):

    create_folders_if_not_exist(['results'])
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    X_T = transform_space(X)

    X_T_train, X_T_test, y_train, y_test = sample_data(X_T, label, RANDOM_SEED, val = False)

    accuracy_lineal, svm_lineal = run_lineal_kernel(X_T_train, y_train, X_T_test, y_test)
    accuracy_rbf, svm_rbf = run_rbf_kernel(X_T_train, y_train, X_T_test, y_test)

    if graph:
        plot_results(svm_lineal, X_T, label, 'lineal', None, exp)
        plot_results(svm_rbf, X_T, label, 'rbf', None, exp)

    best_acc_fisher = 0

    results = []
    for t in ts:
        for q in qs:
            params = dict(zip(['t', 'q'], [t, q]))
            
            accuracy_fisher, svm_fisher = run_fisher_kernel(X_T_train, y_train, X_T_test, y_test, t, q, RANDOM_SEED)
            results.append([t, q, accuracy_fisher, accuracy_lineal, accuracy_rbf])

            if graph:
                plot_results(svm_fisher, X_T, label, 'fisher', params, exp)
        
    df_results = pd.DataFrame(results, columns=['t', 'q', 'fisher', 'lineal', 'rbf'])
    df_results.to_csv(path, index=False)


def transform_data(data):
    X1 = np.mean(data, axis=1)
    X2 = np.std(data, axis=1)
    X = np.concatenate((X1.reshape(-1, 1), X2.reshape(-1, 1)), axis=1)
    return X

def read_data(path):
    paths = []
    for root, _, files in os.walk(path):
        for file in files:
            paths.append(os.path.join(root, file))
    
    n_classes = len(paths)

    
    data = []
    N = 0
    for path in paths:
        df = pd.read_csv(path, header=None)
        data_i = df.to_numpy()
        N = data_i.shape[0]
        data.append(data_i)

    X = np.concatenate(data, axis=0)
    labels = np.array([np.repeat(i, N) for i in range(n_classes)]).flatten()

    return X, labels


def read_generated_data():
    df1 = pd.read_csv('points/exp1/mu0_sigma1_q2.csv')
    data1 = df1.to_numpy()
    df2= pd.read_csv('points/exp1/mu2_sigma2_q2.csv')
    data2 = df2.to_numpy()
    data = np.concatenate((data1, data2), axis=0)
    label = np.concatenate((np.zeros(data1.shape[0]), np.ones(data2.shape[0])))
    return data, label