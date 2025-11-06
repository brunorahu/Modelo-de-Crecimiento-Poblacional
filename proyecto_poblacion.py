import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- 1. Definición de la Solución Analítica ---
# Esta es la función que derivamos (N(t)) [cite: 53]
# 't' es la variable independiente (años)
# 'r' y 'kappa' son los parámetros que queremos encontrar 
def funcion_logistica(t, r, kappa):
    # Usamos los datos iniciales (t0, N0) como constantes conocidas
    # t0 = 1950, N0 = 2471424002 (los tomaremos del archivo)
    N0 = 2471424002.0
    t0 = 1950.0
    
    # Esta es la ecuación analítica que resolvimos
    # N(t) = kappa / (1 + ((kappa - N0) / N0) * exp(-r * (t - t0)))
    try:
        # Usamos np.exp para manejar arrays de numpy
        termino_exp = ((kappa - N0) / N0) * np.exp(-r * (t - t0))
        return kappa / (1 + termino_exp)
    except OverflowError:
        # Manejo de error si los valores se vuelven muy grandes durante el ajuste
        return np.inf

# --- 2. Carga y Preparación de Datos ---
# Cargamos los datos de poblacion.txt [cite: 53, 59]
# El archivo usa un tabulador (\t) como separador
data = pd.read_csv('poblacion.txt', sep='\t', header=None, names=['anio', 'poblacion'])

# Convertimos las columnas a arrays de NumPy para Scipy
# Aseguramos que sean de tipo 'float' para los cálculos
t_datos = np.array(data['anio'], dtype=float)
N_datos = np.array(data['poblacion'], dtype=float)

# --- 3. Ajuste de Curva (curve_fit) ---
# [cite: 55]
# 'curve_fit' necesita la función del modelo, los datos 'x' (t_datos),
# y los datos 'y' (N_datos).

# Valores iniciales (estimados) para r y kappa.
# r ~ 0.01 (crecimiento pequeño)
# kappa ~ 1e10 (un número grande, ej. 10 mil millones, algo mayor que el último dato)
# Esto ayuda al algoritmo a converger más rápido.
valores_iniciales = [0.01, 1e10] 

# 'popt' contendrá los parámetros optimizados (r, kappa)
# 'pcov' contendrá la covarianza estimada
popt, pcov = curve_fit(funcion_logistica, t_datos, N_datos, p0=valores_iniciales)

# Extraemos los parámetros encontrados
r_opt, kappa_opt = popt

# --- 4. Mostrar Resultados del Ajuste ---
# Estos son los valores pedidos en el punto 2 del reporte 
print("--- Parámetros Optimizados (Punto 2) ---")
print(f"Tasa de crecimiento intrínseca (r) = {r_opt}")
print(f"Capacidad de carga (kappa) = {kappa_opt}")
print("------------------------------------------")

# --- 5. Definición de la Ecuación Diferencial (para métodos numéricos) ---
# Esta es la ecuación original: dN/dt = f(t, N)
# Los métodos numéricos necesitan esta función.
def modelo_diferencial(N, t, r, kappa):
    # 't' no se usa en esta ecuación, pero es estándar incluirlo
    return r * N * (1 - N / kappa)

# --- 6. Implementación de Métodos Numéricos (Punto 3) ---

# Parámetros para la simulación numérica
t_inicial = 1950.0
N_inicial = 2471424002.0
t_final = 2300.0
h = 1.0  # Tamaño del paso (1 año)

# Creamos el array de tiempo
t_numerico = np.arange(t_inicial, t_final + h, h)
N_euler = np.zeros(len(t_numerico))
N_rk4 = np.zeros(len(t_numerico))

# Asignamos condiciones iniciales
N_euler[0] = N_inicial
N_rk4[0] = N_inicial

# Pasamos los valores optimizados a los métodos numéricos
r = r_opt
kappa = kappa_opt

# 6.1 Método de Euler
for i in range(1, len(t_numerico)):
    # N(t+h) = N(t) + h * f(t, N)
    f = modelo_diferencial(N_euler[i-1], t_numerico[i-1], r, kappa)
    N_euler[i] = N_euler[i-1] + h * f

# 6.2 Método de Runge-Kutta (RK4)
for i in range(1, len(t_numerico)):
    t_actual = t_numerico[i-1]
    N_actual = N_rk4[i-1]
    
    k1 = h * modelo_diferencial(N_actual, t_actual, r, kappa)
    k2 = h * modelo_diferencial(N_actual + 0.5 * k1, t_actual + 0.5 * h, r, kappa)
    k3 = h * modelo_diferencial(N_actual + 0.5 * k2, t_actual + 0.5 * h, r, kappa)
    k4 = h * modelo_diferencial(N_actual + k3, t_actual + h, r, kappa)
    
    # N(t+h) = N(t) + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
    N_rk4[i] = N_actual + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

print("\n--- Métodos Numéricos (Punto 3) ---")
print("Soluciones de Euler y Runge-Kutta calculadas hasta el año 2300.")
print("-----------------------------------")

# --- 7. Red Neuronal (Punto 4) ---

print("\n--- Red Neuronal (Punto 4) ---")
# 7.1 Preparación de Datos (Escalado)
# Las redes neuronales funcionan mejor con datos normalizados (ej. entre 0 y 1)

# Los scalers de sklearn necesitan datos en formato 2D (ej. [n_muestras, n_caracteristicas])
t_datos_nn = t_datos.reshape(-1, 1)
N_datos_nn = N_datos.reshape(-1, 1)

scaler_t = MinMaxScaler()
scaler_N = MinMaxScaler()

# Ajustamos los scalers a nuestros datos
t_scaled = scaler_t.fit_transform(t_datos_nn)
N_scaled = scaler_N.fit_transform(N_datos_nn)

# 7.2 Definición del Modelo
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(1,))) # Capa de entrada (1 neurona: año)
model.add(Dense(16, activation='relu'))                   # Capa oculta
model.add(Dense(1))                                      # Capa de salida (1 neurona: población)

# 7.3 Compilación
model.compile(optimizer='adam', loss='mean_squared_error')

# 7.4 Entrenamiento
print("Entrenando la red neuronal...")
model.fit(t_scaled, N_scaled, epochs=500, verbose=0) # verbose=0 para no imprimir 500 líneas
print("Entrenamiento completado.")

# 7.5 Predicción
# Creamos el rango de años para la predicción (1950 a 2300)
t_prediccion_nn_rango = np.arange(t_inicial, t_final + 1, h).reshape(-1, 1)

# Escalamos estos años de entrada
t_prediccion_scaled = scaler_t.transform(t_prediccion_nn_rango)

# Hacemos la predicción con la red
N_prediccion_scaled = model.predict(t_prediccion_scaled)

# Revertimos el escalado para obtener los valores de población reales
N_prediccion_nn = scaler_N.inverse_transform(N_prediccion_scaled)

print("-----------------------------------")


# --- 8. Gráfica Final (Punto 5) ---
# Esta gráfica combina todos los resultados en una sola.

print("\nGenerando gráfica final...")

# Generamos puntos para la solución analítica hasta 2300
t_analitico_proyeccion = np.linspace(t_inicial, t_final, 500)
N_analitico_proyeccion = funcion_logistica(t_analitico_proyeccion, r_opt, kappa_opt)

plt.figure(figsize=(12, 8))

# a) Los datos del archivo poblacion.txt [cite: 52]
plt.plot(t_datos, N_datos, 'o', label='Datos (poblacion.txt)', markersize=4, color='blue')

# b) La solución analítica [cite: 53]
plt.plot(t_analitico_proyeccion, N_analitico_proyeccion, 
         label=f'Solución Analítica (Ajuste)', 
         color='green', linewidth=2.5)

# d) Las soluciones numéricas [cite: 55]
plt.plot(t_numerico, N_euler, '--', label='Sol. Numérica: Euler', 
         color='darkorange', alpha=0.8)
plt.plot(t_numerico, N_rk4, ':', label='Sol. Numérica: Runge-Kutta (RK4)', 
         color='purple', linewidth=2.5, alpha=0.9)

# c) La predicción de red [cite: 54]
plt.plot(t_prediccion_nn_rango, N_prediccion_nn, '.', 
         label='Predicción Red Neuronal', 
         color='red', markersize=2.5, alpha=0.7)

plt.title('Proyección de Población Mundial (Comparativa de Modelos)')
plt.xlabel('Año')
plt.ylabel('Población')
plt.xlim(t_inicial, t_final) # Eje X hasta 2300
plt.ylim(bottom=0) # Que la población empiece en 0
plt.legend()
plt.grid(True)

# Guardar la gráfica para el reporte de LaTeX
plt.savefig('grafica_resultados.png', dpi=300)
print(f"Gráfica final guardada como 'grafica_resultados.png'")

plt.show()