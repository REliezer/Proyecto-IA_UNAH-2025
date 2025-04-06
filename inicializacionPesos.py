import numpy as np 

#Glorot/Xavier Uniform and Normal
class InicializacionPesos:
    @staticmethod
    # For the normal distribution the limit value is constructed by averaging the F-in and F-out together and then taking the square-root.
    # A zero-center (µ = 0) is then used:
    def inicializacion_normal(input, output):
        F_in = input
        F_out = output
        limit = np.sqrt(2 / float(F_in + F_out))
        W = np.random.normal(0.0, limit, size=(F_in, F_out))
        return W
    
    @staticmethod
    # Glorot/Xavier initialization can also be done with a uniform distribution where we place stronger restrictions on limit:
    def inicializacion_uniforme(input, output):
        F_in = input
        F_out = output
        limit = np.sqrt(6 / float(F_in + F_out))
        W = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))
        return W