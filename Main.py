import numpy as np

import Movimientos as moves
import GraficarMovimientos as gm
import Perceptron as modulo_perceptron

def analisis_resultados(mi_perceptron: modulo_perceptron.Perceptron, iteraciones):
    
    resultado = zip(mi_perceptron.movimientos_identificados, mi_perceptron.movimientos_reales_efectuados)
    contador = 1
    aciertos = 0

    print(f"\nMovimiento identificado - Movimiento real efectuado: ")
    for movimiento_identificado, movimiento_real_efectuado in resultado:
        print(f"Iteracion numero {contador}: {movimiento_identificado} - {movimiento_real_efectuado}")
        if movimiento_identificado == movimiento_real_efectuado:
            aciertos += 1
        contador += 1

    total_movimientos_identificados = len(mi_perceptron.movimientos_identificados)

    precision = aciertos / total_movimientos_identificados

    print(f"\n-> Total de movimientos identificados: {total_movimientos_identificados}")
    print(f"-> Pruebas por movimiento: {iteraciones}")
    print(f'-> Acertados: {aciertos}')
    print(f"-> Porcentaje de aciertos: {precision * 100:.2f}%")


def main():

    neuronas_capa_entrada = 10
    neuronas_capa_oculta = 5
    neuronas_capa_salida = 3

    cantidad_ejemplos = 30

    try: 

        # ------------------ GENERACION DE MOVIMIENTOS ------------------

        mi_perceptron = modulo_perceptron.Perceptron(neuronas_capa_entrada, neuronas_capa_oculta, neuronas_capa_salida)

        print(f"\nPesos capa oculta:")
        for linea in mi_perceptron.pesos_capa_oculta:
            print(linea)

        print(f"\nPesos capa salida:")
        for linea in mi_perceptron.pesos_capa_salida:
            print(linea)

        movimiento_lineal = moves.generar_movimientos_lineales(cantidad_ejemplos)
        movimiento_circular = moves.generar_movimientos_circulares(cantidad_ejemplos)
        movimiento_aleatorio = moves.generar_movimientos_aleatorios(cantidad_ejemplos)
        
        mostrar = input("\nDesea mostrar los movimientos generados? (S/n): ")
        
        if mostrar == 'S' or mostrar == 's':
            gm.plot_movimientos(movimiento_lineal)
            gm.plot_movimiento_circular(movimiento_circular)
            gm.plot_movimientos(movimiento_aleatorio)
        
        # ------------------ IDENTIFICACION DE MOVIMIENTOS ------------------

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

        analisis_resultados(mi_perceptron, cantidad_ejemplos)

    except Exception as e:
        print(e)


main()


