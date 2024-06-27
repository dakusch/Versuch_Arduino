import numpy as np
import matplotlib.pyplot as plt

# Aktivierungsfunktionen
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def linear(x):
    return x

x = np.linspace(-10, 10, 1000)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot-Einstellungen
plot_params = {
    'linewidth': 2.5,
    'alpha': 0.85
}

# Sigmoid
axs[0, 0].plot(x, sigmoid(x), label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', **plot_params)
axs[0, 0].set_title('Sigmoid')
axs[0, 0].legend()
axs[0, 0].set_xticks(np.arange(-10, 11, 2))
axs[0, 0].set_yticks(np.arange(0, 1.1, 0.2))

# Tanh
axs[0, 1].plot(x, tanh(x), label=r'$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$', **plot_params)
axs[0, 1].set_title('Tanh')
axs[0, 1].legend()
axs[0, 1].set_xticks(np.arange(-10, 11, 2))
axs[0, 1].set_yticks(np.arange(-1, 1.1, 0.5))

# ReLU
axs[1, 0].plot(x, relu(x), label=r'$\mathrm{ReLU}(z) = \max(0, z)$', **plot_params)
axs[1, 0].set_title('ReLU')
axs[1, 0].legend()
axs[1, 0].set_xticks(np.arange(-10, 11, 2))
axs[1, 0].set_yticks(np.arange(0, 11, 2))

# Linear
axs[1, 1].plot(x, linear(x), label=r'$f(z) = z$', **plot_params)
axs[1, 1].set_title('Linear')
axs[1, 1].legend()
axs[1, 1].set_xticks(np.arange(-10, 11, 2))
axs[1, 1].set_yticks(np.arange(-10, 11, 2))

# Layout und Anzeigen
plt.tight_layout()
plt.savefig('Versuch_Arduino_Skript/images/activation_functions.pdf')
