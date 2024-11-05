import numpy as np

def generar_movimientos_lineales(iteraciones) -> list:

    movimientos_generados = [] 

    for _ in range (iteraciones):

        # (x0, y0) -> Posicion inicial
        x0 = np.random.uniform(-5, 5)
        y0 = np.random.uniform(-5, 5)
        
        # (dx, dy) -> representa el desplzamiento
        dx = np.random.uniform(-1,1)
        dy = np.random.uniform(-1,1)

        movimiento_temporal = []

        for step in range(10): # 10 'pasos' en el tiempo
            x1 = x0 + dx
            y1 = y0 + dy
            movimiento_temporal.append([x1, y1])

            # Actualizar la posicion al final del paso
            x0 = x1 
            y0 = y1

        movimientos_generados.append(movimiento_temporal)

    return movimientos_generados

def generar_movimientos_circulares(iteraciones) -> list:

    movimientos_producidos = [] 

    for _ in range (iteraciones):

        # (cx, cy) -> Posicion del centro
        cx = np.random.uniform(-5, 5)
        cy = np.random.uniform(-5, 5)
        
        # radio -> Radio del circulo
        radio = np.random.uniform(1, 3)

        movimiento_temporal = []

        for angulo in range(10):

            '''
                2 * np.pi -> Angulo en radianes
                2 * np.pi * angulo / 10 -> divide el círculo en 10 partes iguales, generando puntos equidistantes a lo largo del perímetro del círculo.
                np.cos(2 * np.pi * angulo/10 ) y np.cos(2 * np.pi * angulo/10 ) -> genera coordenadas x,y en un circulo unitario.
                radio * np.cos y radio * np.sin -> escala el círculo unitario al radio deseado.
                
            '''
            x = cx + radio * np.cos(2 * np.pi * angulo/10 )
            y = cy + radio * np.sin(2 * np.pi * angulo/10 )
            movimiento_temporal.append([x, y])

        movimientos_producidos.append(movimiento_temporal)

    return movimientos_producidos

def generar_movimientos_aleatorios(iteraciones) -> list:

    movimientos_generados = []

    for _ in range (iteraciones):

        movimiento_temporal = []

        # (x0, y0) -> Primer punto aleatorio
        x0 = np.random.uniform(-5, 5)
        y0 = np.random.uniform(-5, 5)

        movimiento_temporal.append([x0, y0])

        for _ in range (9):
            # (dx, dy) -> Desplazamiento aleatorio
            dx = np.random.uniform(-1, 1)
            dy = np.random.uniform(-1, 1)

            x1 = movimiento_temporal[-1][0] + dx # Avanzar en x
            y1 = movimiento_temporal[-1][1] + dy # Avanzar en y

            movimiento_temporal.append([x1, y1])

        movimientos_generados.append(movimiento_temporal)

    return movimientos_generados