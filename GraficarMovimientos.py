import matplotlib.pyplot as plt

def plot_movimientos(movimientos: list):
    for movimiento in movimientos:
        x = [punto[0] for punto in movimiento]
        y = [punto[1] for punto in movimiento]
        plt.plot(x, y, marker='o')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Grafica")
    plt.grid(True)
    plt.show()

def plot_movimiento_circular(movimientos: list):
    for movimiento in movimientos:
        x = [punto[0] for punto in movimiento]
        y = [punto[1] for punto in movimiento]
        line, = plt.plot(x, y, marker='o')
        
        # Conectar el primer y el último punto con el mismo estilo de línea
        plt.plot([x[0], x[-1]], [y[0], y[-1]], color=line.get_color(), marker='o')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Grafica con Conexión")
    plt.grid(True)
    plt.show()



