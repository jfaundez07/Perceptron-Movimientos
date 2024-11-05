import numpy as np

class Perceptron:

    def __init__(self, capa_entrada: int, capa_oculta: int, capa_salida: int):
        self.pesos_capa_oculta = generar_pesos(capa_entrada, capa_oculta) # Matriz de pesos desde la capa de entrada a la capa oculta
        self.pesos_capa_salida = generar_pesos(capa_oculta, capa_salida) # Matriz de pesos desde la capa oculta a la capa de salida
        self.movimientos_identificados = []
        self.movimientos_reales_efectuados = []
    
    def propagacion_hacia_adelante(self, datos_entrada: np.array) -> np.array:
        '''
            Se realiza la multiplicacion de la matriz de pesos de la capa de entrada con la capa oculta.
            Se realiza la multiplicacion de la matriz de pesos de la capa oculta con la capa de salida.
        '''
        capa_entrada = np.array(datos_entrada).flatten()[:10]
        capa_oculta = funcion_escalon(np.dot(capa_entrada, self.pesos_capa_oculta))
        capa_oculta = capa_oculta[np.newaxis, :] # Se agrega una nueva dimensión a la matriz para que siempore sea un arreglo bideimensional
        capa_salida = funcion_softmax(np.dot(capa_oculta, self.pesos_capa_salida))
        return capa_salida
    
    def clasificacion_movimiento(self, probabilidades: np.array):
        opciones = ['Lineal', 'Circular', 'Aleatorio']
        maxima_probabilidad_opciones = np.argmax(probabilidades) # Se obtiene el índice de la máxima probabilidad
        return opciones[maxima_probabilidad_opciones]
    
# ------------------- Funciones auxiliares -------------------

def generar_pesos(filas: int, columnas: int) -> np.array: 
    '''
        Se asocian la cantidad de neuronas de entrada y salida, con una matriz de pesos aleatorios.
        n neuronas que reciben la informacion = n filas
        n neuronas que entregan la informacion = n columnas
    '''
    return np.random.rand(filas, columnas) 

def funcion_escalon(input_x: np.array) -> np.array:
    return np.where(input_x >= 0, 1, 0)

def funcion_softmax(input_x: np.array) -> np.array:
    '''
        np.max(): calcula el valor máximo a lo largo de cada fila de input_x
        axis=1: se realiza la operación a lo largo de las filas
        keepdims=True: mantiene las dimensiones de input_x
    '''
    exponenciales_x = np.exp(input_x - np.max(input_x, axis=1, keepdims=True))
    return exponenciales_x / np.sum(exponenciales_x, axis=1, keepdims=True) 

