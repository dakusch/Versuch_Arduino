#plot binary data and the moving average over one period
import matplotlib.pyplot as plt
import numpy as np

# Zeitachse
t = np.linspace(0, 1, 1000)

# Erstellen der binären Daten, die moduliert werden (1 und 0 level)
binary_data = np.signbit(np.sin(2 * np.pi * 2 * t+np.pi))

# Moving Average
window_size = 250
buffer = np.zeros(window_size, dtype=np.float32)
moving_average = np.zeros(len(binary_data))
for i in range(len(binary_data)):
    buffer[:-1] = buffer[1:]
    buffer[-1] = binary_data[i]
    moving_average[i] = np.mean(buffer)

# Verbesserte und untereinander angeordnete Plots
plt.figure(figsize=(10, 6.66))

# Binäre Daten Plot
plt.subplot(2, 1, 1)
plt.plot(t, binary_data, 'r')
plt.title('Binäre Daten')
plt.xlabel('Zeit in ms')
plt.ylabel('Amplitude')
plt.ylim(-1.5, 1.5)
plt.grid(True)

# Moving Average Plot
plt.subplot(2, 1, 2)
plt.plot(t, moving_average, 'b')
plt.title('Gleitender Mittelwert über Dauer eines Symbols')
plt.xlabel('Zeit in ms')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()