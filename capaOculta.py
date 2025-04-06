import numpy as np
from inicializacionPesos import InicializacionPesos as iniPesos

class CapaOculta:
    """
    Constructor de la capa oculta, inicializa pesos y sesgos, guarda funciones de activación.
    
    Parámetros:
    - entrada: Número de neuronas que recibe.
    - salida: Número de neuronas que va a tener la capa.
    - activacion: Una función de activación (ReLU, Sigmoid, etc.).
    - derivada_activacion: Derivadada de la función de activación usada.
    
    Atributos:
    - entrada: Número de neuronas que recibe.
    - salida: Número de neuronas que va a tener la capa.
    - activacion: Una función de activación (ReLU, Sigmoid, etc.).
    - derivada_activacion: Derivadada de la función de activación usada.
    - W: Inicialización de pesos.
    - sesgos: Inicialización de sesgos (bias).
    - entrada: Para guardar los datos intermedios durante forward() y backward().
    - salida: Para guardar los datos intermedios durante forward() y backward().
    """
    def __init__(self, entrada, salida, activacion, derivada_activacion):
        self.entrada = entrada
        self.salida = salida
        self.activacion = activacion
        self.derivada_activacion = derivada_activacion

        self.W = iniPesos.inicializacion_uniforme(entrada, salida)
        self.sesgos = np.zeros((1, salida)) #Crear una matriz de ceros de tamaño (1, salida) fila x columna

        # Variables para backpropagation
        self.entrada = None
        self.salida = None

    """
    Propagación hacia adelante.
    
    Parámetros:
    - entrada: Matriz de entradas.
    
    Atributos:
    - entrada: Matriz de entradas.
    - z: Calcula la salida de la capa.
    - salida: Aplica la función de activación.

    Retorna:
    - salida: Salida de la capa.

    """
    def forward(self, entrada):
        self.entrada = entrada  # Guardamos para backpropagation
        z = np.dot(entrada, self.W) + self.sesgos # Producto matricial entre entradas y pesos
        self.salida = self.activacion(z) # Aplicar a z una función de activación para evitar que sea lineal
        return self.salida
    
    """
    Propagación hacia atrás (Backpropagation), ajusta pesos/sesgos y pasa el error hacia atrás (aprendizaje).
    
    Parámetros:
    - gradiente_salida: El gradiente de salida de la capa siguiente.
    - tasa_aprendizaje: Factor que controla la magnitud de la actualización.
    
    Atributos:
    - delta: Calcula el error interno.
    - gradiente_pesos: Calcula los gradientes (para aprender), como ajustar pesos y sesgos para mejorar la predicción.
    - gradiente_sesgos: Calcula los gradientes (para aprender), como ajustar pesos y sesgos para mejorar la predicción.
    - gradiente_entrada: Pasa el error hacia atrás a la capa anterior.
    - W: Actualización de los pesos.
    - sesgos: Actualización de los sesgos (bias).
    
    Retorna:
    - gradiente_entrada: Devuelve el gradiente hacia atrás.
    """
    def backward(self, gradiente_salida, tasa_aprendizaje):
        # Derivada de la activación
        delta = gradiente_salida * self.derivada_activacion(self.salida)

        # Gradientes de los pesos y sesgos
        gradiente_pesos = np.dot(self.entrada.T, delta)
        gradiente_sesgos = np.sum(delta, axis=0, keepdims=True)

        # Gradiente para la capa anterior
        gradiente_entrada = np.dot(delta, self.W.T)

        # Actualizamos pesos y sesgos
        self.W -= tasa_aprendizaje * gradiente_pesos
        self.sesgos -= tasa_aprendizaje * gradiente_sesgos

        return gradiente_entrada