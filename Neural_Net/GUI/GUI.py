import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, PhotoImage, Label
import threading
import time
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import model_functions  # Stelle sicher, dass model_functions.py im selben Verzeichnis liegt


class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Feedforward Netzwerk Trainer")
        self.root.configure(bg='white')
        self.root.geometry("1000x1000")

        self.model = None  # Hält das trainierte Keras-Modell

        self.create_widgets()

    def create_widgets(self):
        # Haupt-Frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Logo
        self.logo_image = PhotoImage(file='images/antlogo.png').subsample(3, 3)
        self.logo_label = Label(self.root, image=self.logo_image, bg='white')
        self.logo_label.grid(row=0, column=4, sticky="ne", padx=10, pady=10)

        # Trainingsdaten Eingabefeld
        ttk.Label(self.main_frame, text="Trainingsdaten:").grid(row=0, column=0, sticky="w")
        self.filepath_entry = ttk.Entry(self.main_frame)
        self.filepath_entry.grid(row=0, column=1, sticky="ew", pady=5)
        ttk.Button(self.main_frame, text="Datei auswählen", command=self.select_file).grid(row=0, column=2, sticky="ew", padx=5)

        # Weitere Eingabefelder und Buttons
        self.create_training_controls()

        # Konsolen-Ausgabe
        self.console_output = scrolledtext.ScrolledText(self.main_frame, height=10)
        self.console_output.grid(row=9, column=0, columnspan=3, sticky="nsew", pady=10)
        self.console_output.configure(state='disabled')  # Verhindere, dass der Benutzer den Text bearbeiten kann

        self.visualize_button = ttk.Button(self.main_frame, text="Netzwerk visualisieren", command=self.visualize_model_neurons)
        self.visualize_button.grid(row=12, column=0, columnspan=3, pady=10, sticky="ew")


        # Anpassungen für das dynamische Grid-Layout
        for col in range(3):
            self.main_frame.grid_columnconfigure(col, weight=1)

    def create_training_controls(self):
        # Testdaten Prozent
        ttk.Label(self.main_frame, text="Testdaten Prozent:").grid(row=1, column=0, sticky="w")
        self.test_data_percentage_entry = ttk.Entry(self.main_frame)
        self.test_data_percentage_entry.insert(0, '0.2')
        self.test_data_percentage_entry.grid(row=1, column=1, sticky="ew", pady=5)

        # Epochen
        ttk.Label(self.main_frame, text="Epochen:").grid(row=2, column=0, sticky="w")
        self.num_epochs_entry = ttk.Entry(self.main_frame)
        self.num_epochs_entry.insert(0, '5')
        self.num_epochs_entry.grid(row=2, column=1, sticky="ew", pady=5)

        # Versteckte Schichten
        ttk.Label(self.main_frame, text="Versteckte Schichten:").grid(row=3, column=0, sticky="w")
        self.hidden_layers_entry = ttk.Entry(self.main_frame)
        self.hidden_layers_entry.insert(0, '2')
        self.hidden_layers_entry.grid(row=3, column=1, sticky="ew", pady=5)

        # Versteckte Einheiten
        ttk.Label(self.main_frame, text="Einheiten pro versteckter Schicht").grid(row=4, column=0, sticky="w")
        self.hidden_units_entry = ttk.Entry(self.main_frame)
        self.hidden_units_entry.insert(0, '2')
        self.hidden_units_entry.grid(row=4, column=1, sticky="ew", pady=5)

        # Dropdown-Menü zur auswahl der Aktivierungsfunktion
        ttk.Label(self.main_frame, text="Aktivierungsfunktion Versteckte Schichten").grid(row=5, column=0, sticky="w")
        self.activation_function_var = tk.StringVar()
        self.activation_function_var.set('relu')
        self.activation_function_dropdown = ttk.Combobox(self.main_frame, textvariable=self.activation_function_var)
        self.activation_function_dropdown['values'] = ['relu', 'sigmoid', 'tanh', 'softmax']
        self.activation_function_dropdown.grid(row=5, column=1, sticky="ew", pady=5)

        # Dropdown-Menü zur auswahl der Aktivierungsfunktion
        ttk.Label(self.main_frame, text="Aktivierungsfunktion letzte Schicht").grid(row=6, column=0, sticky="w")
        self.activation_function_var_last = tk.StringVar()
        self.activation_function_var_last.set('relu')
        self.activation_function_last_dropdown = ttk.Combobox(self.main_frame, textvariable=self.activation_function_var_last)
        self.activation_function_last_dropdown['values'] = ['relu', 'sigmoid', 'tanh', 'softmax']
        self.activation_function_last_dropdown.grid(row=6, column=1, sticky="ew", pady=5)

        # Modellname
        ttk.Label(self.main_frame, text="Modellname:").grid(row=7, column=0, sticky="w")
        self.model_name_entry = ttk.Entry(self.main_frame)
        self.model_name_entry.grid(row=7, column=1, sticky="ew", pady=5)

        # Trainieren Button
        self.train_button = ttk.Button(self.main_frame, text="Training starten", command=self.start_training)
        self.train_button.grid(row=8, column=0, columnspan=3, pady=10, sticky="ew")

        # Modell speichern Button
        self.save_model_button = ttk.Button(self.main_frame, text="Modell speichern", command=self.save_model)
        self.save_model_button.grid(row=11, column=0, columnspan=3, pady=10, sticky="ew")

    def select_file(self):
        filepath = filedialog.askopenfilename()
        self.filepath_entry.delete(0, tk.END)
        self.filepath_entry.insert(0, filepath)

    def start_training(self):
        def train():
            filepath = self.filepath_entry.get()
            test_data_percentage = float(self.test_data_percentage_entry.get())
            num_epochs = int(self.num_epochs_entry.get())
            hidden_layers = int(self.hidden_layers_entry.get())
            hidden_units = int(self.hidden_units_entry.get())
            activation = self.activation_function_var.get()
            activation_last = self.activation_function_var_last.get()

            train_signals, train_labels, test_signals, test_labels = model_functions.prepare_data(filepath, test_data_percentage)
            self.model = model_functions.define_model((train_signals.shape[1],), train_labels.shape[1], hidden_layers, hidden_units, activation, activation_last)

            gui_update_callback = GUIUpdateCallback(self)
            self.model.fit(train_signals, train_labels, epochs=num_epochs, callbacks=[gui_update_callback])

        training_thread = threading.Thread(target=train)
        training_thread.start()

    def update_console(self, msg):
        self.console_output.configure(state='normal')  # Erlaube das Hinzufügen von Text
        self.console_output.insert(tk.END, msg)  # Füge die Nachricht hinzu
        self.console_output.configure(state='disabled')  # Verhindere weitere Bearbeitung
        self.console_output.yview(tk.END)  # Scrolle zum Ende des Textfelds

    def save_model(self):
        if self.model:
            model_name = self.model_name_entry.get()
            if model_name:
                model_path = f"./data/{model_name}.keras"
                self.model.save(model_path)
                messagebox.showinfo("Erfolg", f"Modell wurde erfolgreich gespeichert als {model_path}")
            else:
                messagebox.showwarning("Warnung", "Bitte gib einen Modellnamen ein.")
        else:
            messagebox.showwarning("Warnung", "Kein trainiertes Modell zum Speichern vorhanden.")

    def visualize_model_neurons(self):
        if self.model:  # Stelle sicher, dass ein Modell vorhanden ist
            
            fig = Figure(figsize=(8, 4))
            ax = fig.add_subplot(111)
            G = nx.DiGraph()

                # Initialisiere die Koordinaten für die Knotenplatzierung
            x = 0  # Startposition für die erste Schicht
            layer_width = 2  # Horizontale Distanz zwischen den Schichten
            neuron_spacing = 1  # Vertikale Distanz zwischen den Neuronen innerhalb einer Schicht
            
            pos = {}  # Dictionary zur Speicherung der Positionen der Knoten
            layers_neuron_indices = []  # Speichert die Indizes der Neuronen jeder Schicht
            
            for i, layer in enumerate(self.model.layers):
                layer_neurons = []
                num_neurons = layer.units if hasattr(layer, 'units') else 1  # Anzahl der Neuronen in der Schicht
                y = (num_neurons - 1) * neuron_spacing / 2  # Zentriere die Neuronen vertikal

                for j in range(num_neurons):
                    neuron_index = f"L {i+1}\nN {j+1}"
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


            # Zeichne den Graphen auf die 'ax' Achse
            nx.draw(G, pos, ax=ax, with_labels=True, node_size=400, node_color=(0.204, 0.373, 0.667), font_size=10, font_weight="light", arrows=True)

            # Erstelle ein FigureCanvasTkAgg-Objekt und binde es an das Tkinter-Fenster
            canvas = FigureCanvasTkAgg(fig, master=self.root)  # 'self.root' ist das Tkinter-Hauptfenster
            canvas.draw()
            canvas.get_tk_widget().grid(row=11, column=0, columnspan=3, pady=10, sticky="nsew")

            # Konfiguriere das Grid im Haupt-Frame, um den Canvas aufzunehmen
            self.main_frame.grid_rowconfigure(11, weight=1)
            for col in range(3):
                self.main_frame.grid_columnconfigure(col, weight=1)
        else:
            messagebox.showwarning("Warnung", "Kein Modell zur Visualisierung vorhanden.")


class GUIUpdateCallback(tf.keras.callbacks.Callback):
    def __init__(self, app):
        super().__init__()
        self.app = app

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        message = f"Epoch {epoch + 1}: Genauigkeit = {accuracy:.4f} %\n"
        self.app.root.after(0, self.app.update_console, message)

    def on_train_end(self, logs=None):
        self.app.root.after(0, self.app.update_console, "\nTraining abgeschlossen.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
