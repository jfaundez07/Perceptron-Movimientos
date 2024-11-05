import numpy as np

import Movimientos as moves
import GraficarMovimientos as gm
import Perceptron as perceptron

def main():

    neuronas_capa_entrada = 10
    neuronas_capa_oculta = 5
    neuronas_capa_salida = 3

    cantidad_ejemplos = 30

    try: 

        mi_perceptron = perceptron.Perceptron(neuronas_capa_entrada, neuronas_capa_oculta, neuronas_capa_salida)

        movimiento_lineal = moves.generar_movimientos_lineales(cantidad_ejemplos)
        movimiento_circular = moves.generar_movimientos_circulares(cantidad_ejemplos)
        movimiento_aleatorio = moves.generar_movimientos_aleatorios(cantidad_ejemplos)
        
        mostrar = input("Desea mostrar los movimientos generados? (S/n): ")
        
        if mostrar == 'S' or mostrar == 's':
            gm.plot_movimientos(movimiento_lineal)
            gm.plot_movimiento_circular(movimiento_circular)
            gm.plot_movimientos(movimiento_aleatorio)
        

        for movimiento in movimiento_lineal:

            entrada = np.array(movimiento).flatten()

            probabilidades = mi_perceptron.propagacion_hacia_adelante(entrada)
            movimiento_identificado = mi_perceptron.clasificacion_movimiento(probabilidades)

            mi_perceptron.movimientos_identificados.append(movimiento_identificado)
            mi_perceptron.movimientos_reales_efectuados.append('Lineal')

        for movimiento in movimiento_circular:
                
            entrada = np.array(movimiento).flatten()

            probabilidades = mi_perceptron.propagacion_hacia_adelante(entrada)
            movimiento_identificado = mi_perceptron.clasificacion_movimiento(probabilidades)

            mi_perceptron.movimientos_identificados.append(movimiento_identificado)
            mi_perceptron.movimientos_reales_efectuados.append('Circular')
        
        for movimiento in movimiento_aleatorio:
                    
            entrada = np.array(movimiento).flatten()

            probabilidades = mi_perceptron.propagacion_hacia_adelante(entrada)
            movimiento_identificado = mi_perceptron.clasificacion_movimiento(probabilidades)

            mi_perceptron.movimientos_identificados.append(movimiento_identificado)
            mi_perceptron.movimientos_reales_efectuados.append('Aleatorio')

        # ------------------ ANALISIS FINAL ------------------

        resultado = zip(mi_perceptron.movimientos_identificados, mi_perceptron.movimientos_reales_efectuados)

        print(f"Movimiento identificado - Movimiento real efectuado ")
        contador = 1
        for movimiento_identificado, movimiento_real_efectuado in resultado:
            print(f"Iteracion numero {contador}: {movimiento_identificado} - {movimiento_real_efectuado}")
            contador += 1
            

        aciertos = 0
        total_movimientos_identificados = len(mi_perceptron.movimientos_identificados)

        for movimiento_identificado, movimiento_real_efectuado in zip(mi_perceptron.movimientos_identificados, mi_perceptron.movimientos_reales_efectuados):
            if movimiento_identificado == movimiento_real_efectuado:
                aciertos += 1

        porcentaje_aciertos = aciertos / total_movimientos_identificados

        print(f"Total de movimientos identificados: {total_movimientos_identificados}")
        print(f"Pruebas por movimiento: {cantidad_ejemplos}")
        print (f"Porcentaje de aciertos: {porcentaje_aciertos * 100:.2f}%")

    except Exception as e:
        print(e)


main()


