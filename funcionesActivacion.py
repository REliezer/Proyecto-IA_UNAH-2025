import numpy as np 

class FuncionesActivacion:
    #Sigmoide
    @staticmethod
    def sigmoide(x):
        return 1 / (1 + np.exp(-x))

    #Tangente hiperbólica
    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    #ReLu (Rectified Linear Unit)
    #Retorna 0 para cualquier valor x <= 0 y retorna x para cualquier valor x > 0.
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    #ELU (Exponencial Linear Units)
    @staticmethod
    def elu(alpha):
        def activacion(x):
            return np.where(x > 0, x, alpha * (np.exp(x) - 1))
        return activacion
    
    #Derivadas de las funciones activaciones
    @staticmethod
    def sigmoide_derivada(x):
        s = FuncionesActivacion.sigmoide(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh_derivada(x):
        t = np.tanh(x)
        return 1 - t ** 2 
       
    @staticmethod
    #Devuelve 1 para cualquier x > 0, y 0 en caso contrario
    def relu_derivada(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def elu_derivada(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))

    """
    Función de pérdida (MSE)

    Parámetros:
    - y: Valores reales.
    - y_pred: Valores predichos.

    Retorna:
    - MSE: Error cuadrático medio.
    """
    @staticmethod
    def mse(y, y_pred):
        return np.mean((y - y_pred) ** 2)

    """
    Derivada de la función de pérdida (MSE)

    Parámetros:
    - y: Valores reales.
    - y_pred: Valores predichos.

    Retorna:
    - derivada: Derivada de la función de pérdida.
    """
    @staticmethod
    def mse_derivada(y, y_pred):
        return 2 * (y_pred - y) / y.size