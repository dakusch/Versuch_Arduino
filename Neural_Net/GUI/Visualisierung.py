import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf

def visualize_model_neurons(model):
    G = nx.DiGraph()
    
    # Initialisiere die Koordinaten für die Knotenplatzierung
    x = 0  # Startposition für die erste Schicht
    layer_width = 2  # Horizontale Distanz zwischen den Schichten
    neuron_spacing = 1  # Vertikale Distanz zwischen den Neuronen innerhalb einer Schicht
    
    pos = {}  # Dictionary zur Speicherung der Positionen der Knoten
    layers_neuron_indices = []  # Speichert die Indizes der Neuronen jeder Schicht
    
    for i, layer in enumerate(model.layers):
        layer_neurons = []
        num_neurons = layer.units if hasattr(layer, 'units') else 1  # Anzahl der Neuronen in der Schicht
        y = (num_neurons - 1) * neuron_spacing / 2  # Zentriere die Neuronen vertikal

        for j in range(num_neurons):
            neuron_index = f"Schicht {i+1}\nNeuron {j+1}"
            G.add_node(neuron_index)
            layer_neurons.append(neuron_index)
            
            # Setze die Position für jeden Neuronen-Knoten
            pos[neuron_index] = (x, -y)  # Negative y-Werte, da höhere Werte nach unten gehen
            y -= neuron_spacing  # Aktualisiere die y-Position für das nächste Neuron

        layers_neuron_indices.append(layer_neurons)
        x += layer_width  # Verschiebe die x-Position für die nächste Schicht

    # Verbinde die Neuronen zwischen den Schichten
    for i in range(len(layers_neuron_indices) - 1):
        for src in layers_neuron_indices[i]:
            for dst in layers_neuron_indices[i + 1]:
                G.add_edge(src, dst)

    # Zeichne den Graphen
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
    plt.title("Neuronale Netzwerk Visualisierung")
    plt.axis('on')  # Schaltet die Achsen ein, um die Schichtanordnung besser zu visualisieren
    plt.show()

# Beispiel: Ein einfaches Keras-Modell definieren
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(10,), activation='relu', name='Eingabe'),
    tf.keras.layers.Dense(10, activation='relu', name='Versteckt'),
    tf.keras.layers.Dense(1, activation='sigmoid', name='Ausgabe')
])

# Visualisiere das Modell mit Neuronen
visualize_model_neurons(model)
