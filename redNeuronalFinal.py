import numpy as np
from funcionesActivacion import FuncionesActivacion as funcAct

class RedNeuronal:
    """
    Constructor de la clase. Se inicializa la red neuronal con una lista de capas ocultas.

    Parámetros:
    - capas: Lista de objetos tipo CapaOculta.

    """
    def __init__(self, capas):
        self.capas = capas  # Lista de objetos tipo CapaOculta

    """
    Se raliza el paso hacia adelante (forward pass) a través de la red neuronal, procesando las entradas
    X a través de todas las capas de la red.

    Parámetros:
    - X: Caracterisiticas.

    Retorna:
    - salida: Resultado después de pasar por todas las capas.
    """
    def forward(self, X):
        salida = X
        for capa in self.capas:
            salida = capa.forward(salida)
        return salida

    """
    Método de retropropagación (backpropagation), que actualiza los pesos de las capas para minimizar el error.

    Parámetros:
    - gradiente_salida: El gradiente de salida de la capa siguiente.
    - tasa_aprendizaje: La tasa de aprendizaje que controla cuán grandes son los ajustes en los pesos.

    """
    def backward(self, gradiente_salida, tasa_aprendizaje):
        # Calcula el gradiente inicial desde la función de pérdida
        gradiente = gradiente_salida

        # Retropropagación a través de todas las capas (de última a primera)
        for capa in reversed(self.capas): 
            gradiente = capa.backward(gradiente, tasa_aprendizaje)

    """
    Método para entrenar la red neuronal. Realiza el proceso de entrenamiento durante un número de épocas (epochs).

    Parámetros:
    - X: Características o datos de entrada de entrenamiento.
    - y: Son los valores reales que queremos que la red prediga.
    - epochs: Número de iteraciones o épocas que la red realizará durante el entrenamiento.
    - tasa_aprendizaje: La tasa de aprendizaje que se usa para actualizar los pesos.

    """
    def entrenar(self, X, y, epochs, tasa_aprendizaje):
        for epoca in range(epochs):
            y_pred = self.forward(X) # Realiza el forward pass
            perdida = funcAct.mse(y, y_pred) # Calcula la pérdida (error)
            grad = funcAct.mse_derivada(y, y_pred) # Calcula la derivada de la pérdida
            self.backward(grad, tasa_aprendizaje) # Aplica la retropropagación

            if epoca % 100 == 0 or epoca == epochs - 1:
                print(f"Época {epoca + 1}/{epochs} - MSE: {perdida:.6f}")

    """
    Para hacer predicciones.

    Parámetros:
    - X: Conjunto de datos de prueba.

    Retorna:
    - salida: Resultado de la predicción.

    """
    def predecir(self, X):
        return self.forward(X)
