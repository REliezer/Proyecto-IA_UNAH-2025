import cv2
import numpy as np

class PreprocesarImagen:
    def cargar_imagen(path):
        imagen = cv2.imread(path) # Cargar la imagen
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) # Convertir a escala de grises
        imagen = cv2.resize(imagen, (600, 800)) # Redimensionar la imagen
        imagen = np.array(imagen, dtype=np.float32) / 255.0 # Normalizar entre 0 y 1
        imagen = imagen.flatten() # Convertir a arreglo
        
        # Mostrar la imagen
        #cv2.imshow("Imagen", imagen)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return imagen
