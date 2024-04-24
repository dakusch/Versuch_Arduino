import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from keras.utils import plot_model


# Schritt 1: Datenvorbereitung
def prepare_data(filepath, test_data_percentage):
    data = np.load(filepath)
    signals = data['signals']
    labels = data['labels']

    split_index = int(test_data_percentage * len(signals))
    test_signals, test_labels = signals[:split_index], labels[:split_index]
    train_signals, train_labels = signals[split_index:], labels[split_index:]

    return train_signals, train_labels, test_signals, test_labels

# Schritt 2: Modelldefinition
def define_model(input_shape, output_shape, hidden_layers=2, hidden_units=2, activation='relu', activation_last='sigmoid'):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(hidden_units, activation=activation)] + 
        [tf.keras.layers.Dense(hidden_units, activation=activation) for _ in range(hidden_layers-1)] +
        [tf.keras.layers.Dense(output_shape, activation=activation_last)])  # Sigmoid-Aktivierungsfunktion für binäre Klassifikation
    
    plot_model(model, to_file='images/model_structure.png', show_shapes=True, show_layer_names=True)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    return model

# Schritt 3: Trainingsprozess
def train_model(model, train_signals, train_labels, num_epochs):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_signals, train_labels, epochs=num_epochs, batch_size=32, callbacks=[tensorboard_callback])

# Schritt 4: Evaluation
def evaluate_model(model, test_signals, test_labels):
    test_loss, test_acc = model.evaluate(test_signals, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    return test_loss, test_acc

# Modell speichern
def save_model(model, filepath):
    model.save(filepath)

# Modell Zusammenfassung
def summarize_model(model):
    model.summary()



# Beispielverwendung der Funktionen
if __name__ == "__main__":
    filepath = 'data/ook_dataset.npz'
    test_data_percentage = 0.2
    num_epochs = 5

    train_signals, train_labels, test_signals, test_labels = prepare_data(filepath, test_data_percentage)
    
    if train_signals.shape[1] > 0 and train_labels.shape[1] > 0:
        model = define_model((train_signals.shape[1],), train_labels.shape[1], 2, 2)
        train_model(model, train_signals, train_labels, num_epochs)
        evaluate_model(model, test_signals, test_labels)
        save_model(model, 'data/ook_net_tf.keras')
        summarize_model(model)



