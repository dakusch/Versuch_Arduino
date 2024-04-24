import matplotlib.pyplot as plt
import numpy as np

# Zeitachse
t = np.linspace(0, 1, 1000)

# ASK Signal
ask_signal = (1 + np.sign(np.sin(2 * np.pi * 2 * t))) * np.cos(2 * np.pi * 10 * t)

# FSK Signal
fsk_signal1 = np.sin(2 * np.pi * 10 * t) * (np.sign(np.sin(2 * np.pi * 2 * t)) > 0)
fsk_signal2 = np.sin(2 * np.pi * 15 * t) * (np.sign(np.sin(2 * np.pi * 2 * t)) <= 0)
fsk_signal = fsk_signal1 + fsk_signal2

# Erstellen der binären Daten, die moduliert werden
binary_data = np.signbit(np.sin(2 * np.pi * 2 * t+np.pi))

# Verbesserte und untereinander angeordnete Plots
plt.figure(figsize=(10, 10))

# ASK Plot
plt.subplot(3, 1, 3)
plt.plot(t, ask_signal, 'b')
plt.title('ASK Modulation')
plt.xlabel('Zeit in ms')
plt.ylabel('Amplitude')
plt.grid(True)

# FSK Plot
plt.subplot(3, 1, 2)
plt.plot(t, fsk_signal, 'g')
plt.title('FSK Modulation')
plt.xlabel('Zeit in ms')
plt.ylabel('Amplitude')
plt.grid(True)

# Binäre Daten Plot
plt.subplot(3, 1, 1)
plt.plot(t, binary_data, 'r')
plt.title('Binäre Daten')
plt.xlabel('Zeit in ms')
plt.ylabel('Amplitude')
plt.ylim(-1.5, 1.5)
plt.grid(True)

plt.tight_layout()
plt.savefig('Versuch_Arduino_Skript/images/ASK_FSK.pdf')