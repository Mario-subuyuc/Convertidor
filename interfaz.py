import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import messagebox

# ENTRENAMIENTO DEL MODELO
celsius = np.array([
    -40, -30, -20, -10, -5, 0, 5, 8, 10, 15, 20, 22, 25, 30, 35, 38, 40, 45, 50, 60, 70, 80, 90, 100
], dtype=float)

fahren = np.array([
    -40, -22, -4, 14, 23, 32, 41, 46.4, 50, 59, 68, 71.6, 77, 86, 95, 100.4, 104, 113, 122, 140, 158, 176, 194, 212
], dtype=float)


capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')
historial =  modelo.fit(celsius, fahren, epochs=2000, verbose=True)

# FUNCION PARA USAR EL MODELO Y COMPARAR
def predecir():
    try:
        grados = float(entrada.get())
        prediccion = modelo.predict(np.array([grados]))[0][0]
        real = (grados * 9/5) + 32 #obtener valor por formula para comprar con valor de IA
        diferencia = abs(prediccion - real)
        acierto = 100 - (diferencia / real * 100) if real != 0 else 100  # Evitar división por cero

        print(f"IA: {prediccion:.2f}, Real: {real:.2f}, Diferencia: {diferencia:.2f}, Acierto: {acierto:.2f}%")
        print("Variables internas del modelo")
        print(f"Variables internas: {capa.get_weights()}")

        # Cambiar el color dependiendo de la precisión
        if acierto >= 90:
            color = "green"
        elif acierto >= 75:
            color = "orange"
        else:
            color = "red"

        etiqueta_resultado.config(
            text=(
                f"IA: {grados}°C = {prediccion:.2f}°F\n\n"
                f"Fórmula real: {real:.2f}°F\n"
                f"Diferencia: {diferencia:.2f}°F\n"
                f"Precisión: {acierto:.2f}%"
            ),
            fg=color  # Cambiar color del texto según precisión
        )
    except ValueError:
        messagebox.showerror("Error", "Por favor ingresa un número válido.")


import matplotlib.pyplot as plt
plt.xlabel("# Iteracion")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()

# INTERFAZ GRÁFICA
ventana = tk.Tk()
ventana.title("Celsius a Fahrenheit con IA")
ventana.geometry("350x250")

tk.Label(ventana, text="Ingresa grados Celsius:").pack(pady=10)
entrada = tk.Entry(ventana)
entrada.pack()

tk.Button(ventana, text="Convertir", command=predecir).pack(pady=10)

etiqueta_resultado = tk.Label(ventana, text="", justify="left", font=("Arial", 10), fg="black")
etiqueta_resultado.pack(pady=10)

ventana.mainloop()

