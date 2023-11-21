# required libreries
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.svm import SVC
import scipy.integrate as spi
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay

def normalize_min_max(matrix):
    """
    Método para normalizar los datos

    Args: 
        matrix: Matriz de datos original

    Returns: 
        normalized_data: Matriz de datos normalizados
    """
    max_values = np.max(matrix, axis=0)
    min_values = np.min(matrix, axis=0)

    normalized_data = (matrix - min_values) / (max_values - min_values)
    return normalized_data

class FisherKernel():
    def __init__(self, t, q=1) -> None:
        """
        Inicializa la clase FisherKernel con los parámetros t y q.

        Args:
            t: Parámetro t del kernel.
            q: Parámetro q del kernel, por defecto es 1.
        """
        self.t = t
        self.q = q
        const = 1 / (2 ** (5 / 2) + np.pi ** (3 / 2) * self.t ** (-3 / 2))
        self.expr = const * np.exp(-self.t / 4)

    def distH(self, X, Y):
        """
        Calcula la matriz de distancias h entre dos conjuntos de datos.

        Args:
            X: Matriz de datos de forma (n_samples_X, 2).
            Y: Matriz de datos de forma (n_samples_Y, 2).

        Returns:
            Matriz de distancias h de forma (n_samples_X, n_samples_Y).
        """
        rho = np.zeros((X.shape[0], Y.shape[0]))
        for idx_x, xi in enumerate(X):
            for idx_y, yi in enumerate(Y):
                rho[idx_x, idx_y] = np.arccosh(1 + np.sum((xi - yi) ** 2) / (2 * xi[1] * yi[1]))
        return rho

    def to_integrate(self, s, rho):
        """
        Función a integrar en el cálculo del kernel de Fisher.

        Args:
            s: Variable de integración.
            rho: Valor de la distancia h entre dos puntos.

        Returns:
            Resultado de la función a integrar.
        """
        return (s * np.exp(-(s ** 2) / (4 * self.t))) / (np.sqrt(np.cosh(s) - np.cosh(rho)))

    def solve_integral(self, rhos):
        """
        Resuelve la integral numérica para calcular el kernel de Fisher.

        Args:
            rhos: Matriz de distancias h.

        Returns:
            Matriz de kernel de Fisher.
        """
        filas_rho, columnas_rho = rhos.shape
        K = np.zeros((filas_rho, columnas_rho))
        b = np.inf

        for fila in tqdm(range(filas_rho)):
            for columna in range(columnas_rho):
                result, _ = spi.quad(self.to_integrate, rhos[fila, columna], b, args=(rhos[fila, columna],))
                K[fila, columna] = result

        return K

    def kernelFisher(self, X, Y):
        """
        Calcula el kernel de Fisher ajustando los datos de entrada.

        Args:
            X: Matriz de datos de forma (n_samples_X, n_features).
            Y: Matriz de datos de forma (n_samples_Y, n_features).

        Returns:
            Matriz de kernel de Fisher ajustada.
        """
        X[:, 1] = np.sqrt(3 - self.q) * X[:, 1]
        self.t = self.q / (3 - self.q) * self.t
        return self.Kg_H(X, Y)

    def Kg_H(self, X, Y):
        """
        Calcula el kernel de Fisher entre dos conjuntos de datos.

        Args:
            X: Matriz de datos de forma (n_samples_X, n_features).
            Y: Matriz de datos de forma (n_samples_Y, n_features).

        Returns:
            Matriz de kernel de Fisher entre X e Y.
        """
        rhos = self.distH(X, Y)
        K = self.solve_integral(rhos)
        return self.expr * K

    

def plot_results(clf, X, y, kernel_name, params, exp):
    """
    Genera un gráfico que muestra la frontera de decisión y los puntos de entrenamiento
    para un clasificador SVM con un kernel específico.

    Args:
        clf: Clasificador entrenado.
        X: Matriz de datos de forma (n_samples, n_features).
        y: Vector de etiquetas de forma (n_samples,).
        kernel_name: Nombre del kernel utilizado en el SVM.
        params: Parámetros del kernel (opcional).
        exp: Nombre de la carpeta de experimentos donde se guardarán los resultados.

    Returns:
        None
    """
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

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")

    plt.title(title)
    plt.axis("tight")
    plt.savefig(path)

def create_folders_if_not_exist(folder_paths):
    """
    Function that creates folders if they don't exist

    Args:
        folder_paths -> list : List with the paths of the folders to create
    """
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

def sample_data(X, label, RANDOM_SEED, val=False):
    """
    Divide los datos en conjuntos de entrenamiento y prueba (y opcionalmente, conjunto de validación).

    Args:
        X: Matriz de datos de forma (n_samples, n_features).
        label: Vector de etiquetas de forma (n_samples,).
        RANDOM_SEED: Semilla para la reproducibilidad de la división de los datos.
        val: Indicador para incluir un conjunto de validación (por defecto es False).

    Returns:
        Tuple: Dependiendo del valor de 'val', devuelve una tupla con los conjuntos de datos y etiquetas:
        - Sin conjunto de validación: (X_train, X_test, y_train, y_test)
        - Con conjunto de validación: (X_train, X_test, X_val, y_train, y_test, y_val)
    """
    if val:
        X_train, X_tv, y_train, y_tv = train_test_split(X, label, test_size=0.4, random_state=RANDOM_SEED)
        X_test, X_val, y_test, y_val = train_test_split(X_tv, y_tv, test_size=0.5, random_state=RANDOM_SEED)

        return X_train, X_test, X_val, y_train, y_test, y_val
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=RANDOM_SEED)

        return X_train, X_test, y_train, y_test

def sample_data_transform(X, label):
    """
    Divide los datos en conjuntos de entrenamiento, prueba y validación, y transforma el espacio de características.

    Args:
        X: Matriz de datos de forma (n_samples, n_features).
        label: Vector de etiquetas de forma (n_samples,).

    Returns:
        Tuple: Devuelve una tupla con los conjuntos de datos y etiquetas originales,
        así como los conjuntos de datos transformados en el espacio de características:
        (X_train, X_test, X_val, y_train, y_test, y_val, X_T_train, X_T_test, X_T_val)
    """
    _, indices_V, indices_T, indices_t = partitioning(X)
    X_T = transform_space(X)

    X_train = X[indices_T, :]
    X_test = X[indices_t, :]
    X_val = X[indices_V, :]

    y_train = label[indices_T]
    y_test = label[indices_t]
    y_val = label[indices_V]

    X_T_train = X_T[indices_T, :]
    X_T_test = X_T[indices_t, :]
    X_T_val = X_T[indices_V, :]

    return X_train, X_test, X_val, y_train, y_test, y_val, X_T_train, X_T_test, X_T_val

def partitioning(datos,size_S = 0.8,size_T= 0.6):
    """
        Método que hace el partitioning de los datos en el set de S, training, testing y validation.

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
    """
    Transforma el espacio de características de una matriz de datos.

    Args:
        X: Matriz de datos de forma (n_samples, n_features).

    Returns:
        T: Matriz transformada de forma (n_samples, 2), donde cada fila contiene
        la media y la desviación estándar de las características correspondientes en X.
    """
    X1 = np.mean(X, axis=1)
    X2 = np.std(X, axis=1)
    T = np.concatenate((X1.reshape(-1, 1), X2.reshape(-1, 1)), axis=1)

    return T

def run_fisher_kernel(X_train, y_train, X_test, y_test, t, q, RANDOM_SEED):
    """
    Entrena un clasificador SVM con un kernel de Fisher personalizado y evalúa su precisión.

    Args:
        X_train: Conjunto de entrenamiento de datos de forma (n_samples_train, n_features).
        y_train: Etiquetas correspondientes al conjunto de entrenamiento de forma (n_samples_train,).
        X_test: Conjunto de prueba de datos de forma (n_samples_test, n_features).
        y_test: Etiquetas correspondientes al conjunto de prueba de forma (n_samples_test,).
        t: Parámetro t para el kernel de Fisher.
        q: Parámetro q para el kernel de Fisher.
        RANDOM_SEED: Semilla para la reproducibilidad del clasificador SVM.

    Returns:
        Tuple: Devuelve una tupla que contiene la precisión del clasificador SVM y la instancia del clasificador.
    """
    instance = FisherKernel(t, q)
    svm_fisher = SVC(kernel=instance.kernelFisher, random_state=RANDOM_SEED)
    svm_fisher.fit(X_train, y_train)
    accuracy_fisher = svm_fisher.score(X_test, y_test)
    fisher_pred = svm_fisher.predict(X_test)

    print('fisher_pred', fisher_pred)
    print("Accuracy con Kernel fisher personalizado: t:", t, " q:", q, accuracy_fisher)

    return accuracy_fisher, svm_fisher

def run_lineal_kernel(X_train, y_train, X_test, y_test):
    """
    Entrena un clasificador SVM con kernel lineal y evalúa su precisión.

    Args:
        X_train: Conjunto de entrenamiento de datos de forma (n_samples_train, n_features).
        y_train: Etiquetas correspondientes al conjunto de entrenamiento de forma (n_samples_train,).
        X_test: Conjunto de prueba de datos de forma (n_samples_test, n_features).
        y_test: Etiquetas correspondientes al conjunto de prueba de forma (n_samples_test,).

    Returns:
        Tuple: Devuelve una tupla que contiene la precisión del clasificador SVM con kernel lineal
        y la instancia del clasificador SVM entrenada.
    """
    svm_lineal = SVC(kernel='linear')
    svm_lineal.fit(X_train, y_train)
    accuracy_lineal = svm_lineal.score(X_test, y_test)

    print("Accuracy con Kernel lineal:", accuracy_lineal)

    return accuracy_lineal, svm_lineal

def run_rbf_kernel(X_train, y_train, X_test, y_test):
    """
    Entrena un clasificador SVM con kernel RBF y evalúa su precisión.

    Args:
        X_train: Conjunto de entrenamiento de datos de forma (n_samples_train, n_features).
        y_train: Etiquetas correspondientes al conjunto de entrenamiento de forma (n_samples_train,).
        X_test: Conjunto de prueba de datos de forma (n_samples_test, n_features).
        y_test: Etiquetas correspondientes al conjunto de prueba de forma (n_samples_test,).

    Returns:
        Tuple: Devuelve una tupla que contiene la precisión del clasificador SVM con kernel RBF
        y la instancia del clasificador SVM entrenada.
    """
    svm_rbf = SVC(kernel='rbf')
    svm_rbf.fit(X_train, y_train)
    accuracy_rbf = svm_rbf.score(X_test, y_test)

    print("Accuracy con Kernel rbf:", accuracy_rbf)

    return accuracy_rbf, svm_rbf

def run_all_configurations(X, label, ts, qs, exp, path, graph):
    """
    Ejecuta las configuraciones de clasificadores SVM con diferentes kernels y parámetros de Fisher.

    Args:
        X: Matriz de datos de forma (n_samples, n_features).
        label: Vector de etiquetas de forma (n_samples,).
        ts: Lista de valores para el parámetro t del kernel de Fisher.
        qs: Lista de valores para el parámetro q del kernel de Fisher.
        exp: Nombre de la carpeta de experimentos donde se guardarán los resultados.
        path: Ruta para guardar los resultados en un archivo CSV.
        graph: Indicador para generar gráficos de resultados.

    Returns:
        None
    """

    create_folders_if_not_exist(['results'])
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    X_T = transform_space(X)
    X_T_train, X_T_test, y_train, y_test = sample_data(X_T, label, RANDOM_SEED, val=False)
    accuracy_lineal, svm_lineal = run_lineal_kernel(X_T_train, y_train, X_T_test, y_test)
    accuracy_rbf, svm_rbf = run_rbf_kernel(X_T_train, y_train, X_T_test, y_test)

    if graph:
        plot_results(svm_lineal, X_T, label, 'lineal', None, exp)
        plot_results(svm_rbf, X_T, label, 'rbf', None, exp)

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
    """
    Transforma los datos al espacio q-gausiano.

    Args:
        data: Matriz de datos de forma (n_samples, n_features).

    Returns:
        X: Matriz transformada de forma (n_samples, 2), donde cada fila contiene
        la media y la desviación estándar de las características correspondientes en data.
    """
    X1 = np.mean(data, axis=1)
    X2 = np.std(data, axis=1)
    X = np.concatenate((X1.reshape(-1, 1), X2.reshape(-1, 1)), axis=1)

    return X

def read_data(path):
    """
    Metodo que lee todos los archivos CSV ubicados en una carpeta

    Args:
        path: Ruta de la carpeta que contiene los archivos CSV.

    Returns:
        Tuple: Devuelve una tupla que contiene una matriz de datos (X) y un vector de etiquetas (labels).
        X tiene forma (n_samples, n_features) y labels tiene forma (n_samples,).
    """
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

def main(ts, qs, experiments, graph):
    """
    Ejecuta las configuraciones de clasificadores SVM con kernels de Fisher para varios experimentos.

    Args:
        ts: Lista de valores para el parámetro t del kernel de Fisher.
        qs: Lista de valores para el parámetro q del kernel de Fisher.
        experiments: Lista de nombres de carpetas que contienen datos para cada experimento.
        graph: Indicador para generar gráficos de resultados.

    Returns:
        None
    """
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
