import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, PhotoImage, Label
import threading
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import model_functions  
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Feedforward Netzwerk Trainer")
        self.root.configure(bg='#2c3e50')
        self.root.geometry("800x700")

        self.style = ttk.Style()
        self.style.configure('TFrame', background='#ecf0f1')
        self.style.configure('TLabel', background='#ecf0f1', font=('Arial', 12))
        self.style.configure('TEntry', font=('Arial', 12))
        self.style.configure('TButton', background='#204373', foreground='white', font=('Arial', 12, 'bold'), relief='flat')
        self.style.map('TButton', relief=[('active', 'groove')])
        self.style.configure('TCombobox', font=('Arial', 12))
        self.style.configure('TScrolledText', font=('Arial', 12))

        self.model = None  # Hält das trainierte Keras-Modell
        self.training_in_progress = False  # Flag, um zu verhindern, dass ein neues Training gestartet wird
        self.canvas_labels = None
        self.canvas = None

        self.mode = tk.StringVar(value="Beginner")

        self.create_widgets()

        # Zeige die anfängliche Grafik mit den Standardparametern an
        self.visualize_model_neurons()

    def create_widgets(self):
        # Haupt-Frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Logo
        self.logo_image = PhotoImage(file='images/antlogo.png').subsample(3, 3)
        self.logo_label = Label(self.main_frame, image=self.logo_image, bg='#2c3e50')
        self.logo_label.grid(row=0, column=1, sticky="ne", padx=10, pady=10)

        # Modus-Auswahl
        self.mode_frame = ttk.Frame(self.main_frame)
        self.mode_frame.grid(row=0, column=0, sticky="ew", pady=5)
        ttk.Label(self.mode_frame, text="Modus:").grid(row=0, column=0, sticky="w")
        self.mode_dropdown = ttk.Combobox(self.mode_frame, textvariable=self.mode, values=["Beginner", "Advanced"], state="readonly")
        self.mode_dropdown.grid(row=0, column=1, sticky="ew", padx=10)
        self.mode_dropdown.bind("<<ComboboxSelected>>", self.toggle_mode)

        # Trainingsdaten Eingabefeld
        self.filepath_frame = ttk.Frame(self.main_frame)
        self.filepath_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(self.filepath_frame, text="Trainingsdaten:").grid(row=0, column=0, sticky="w")
        self.filepath_entry = ttk.Entry(self.filepath_frame, width=50)
        self.filepath_entry.grid(row=0, column=1, sticky="ew", padx=10)
        ttk.Button(self.filepath_frame, text="Datei auswählen", command=self.select_file).grid(row=0, column=2, sticky="ew")

        # Weitere Eingabefelder und Buttons
        self.create_training_controls()

        # Konsolen-Ausgabe
        self.console_output = scrolledtext.ScrolledText(self.main_frame, height=6, wrap=tk.WORD, bg='#bdc3c7', font=('Arial', 12))
        self.console_output.grid(row=15, column=0, columnspan=2, sticky="nsew", pady=10)
        self.console_output.configure(state='disabled')  # Verhindere, dass der Benutzer den Text bearbeiten kann

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(15, weight=1)

    def create_training_controls(self):
        # Testdaten Prozent
        ttk.Label(self.main_frame, text="Testdaten Prozent:").grid(row=2, column=0, sticky="w")
        self.test_data_percentage_entry = ttk.Entry(self.main_frame)
        self.test_data_percentage_entry.insert(0, '20')
        self.test_data_percentage_entry.grid(row=2, column=1, sticky="ew", pady=5)

        # Epochen
        ttk.Label(self.main_frame, text="Epochen:").grid(row=3, column=0, sticky="w")
        self.num_epochs_entry = ttk.Entry(self.main_frame)
        self.num_epochs_entry.insert(0, '1')
        self.num_epochs_entry.grid(row=3, column=1, sticky="ew", pady=5)

        # Versteckte Schichten
        ttk.Label(self.main_frame, text="Versteckte Schichten:").grid(row=4, column=0, sticky="w")
        self.hidden_layers_var = tk.IntVar()
        self.hidden_layers_dropdown = ttk.Combobox(self.main_frame, textvariable=self.hidden_layers_var, state="readonly")
        self.hidden_layers_dropdown['values'] = [i for i in range(1, 6)]
        self.hidden_layers_dropdown.set(2)
        self.hidden_layers_dropdown.grid(row=4, column=1, sticky="ew", pady=5)

        # Versteckte Einheiten
        ttk.Label(self.main_frame, text="Einheiten pro versteckter Schicht").grid(row=5, column=0, sticky="w")
        self.hidden_units_var = tk.IntVar()
        self.hidden_units_dropdown = ttk.Combobox(self.main_frame, textvariable=self.hidden_units_var, state="readonly")
        self.hidden_units_dropdown['values'] = [i for i in range(1, 101)]
        self.hidden_units_dropdown.set(2)
        self.hidden_units_dropdown.grid(row=5, column=1, sticky="ew", pady=5)

        # Dropdown-Menü zur Auswahl der Aktivierungsfunktion
        self.activation_function_label = ttk.Label(self.main_frame, text="Aktivierungsfunktion Versteckte Schichten")
        self.activation_function_label.grid(row=6, column=0, sticky="w")
        self.activation_function_var = tk.StringVar()
        self.activation_function_var.set('relu')
        self.activation_function_dropdown = ttk.Combobox(self.main_frame, textvariable=self.activation_function_var)
        self.activation_function_dropdown['values'] = ['relu', 'sigmoid', 'tanh', 'linear']
        self.activation_function_dropdown.grid(row=6, column=1, sticky="ew", pady=5)

        # Dropdown-Menü zur Auswahl der Aktivierungsfunktion
        self.activation_function_last_label = ttk.Label(self.main_frame, text="Aktivierungsfunktion letzte Schicht")
        self.activation_function_last_label.grid(row=7, column=0, sticky="w")
        self.activation_function_var_last = tk.StringVar()
        self.activation_function_var_last.set('sigmoid')
        self.activation_function_last_dropdown = ttk.Combobox(self.main_frame, textvariable=self.activation_function_var_last)
        self.activation_function_last_dropdown['values'] = ['relu', 'sigmoid', 'tanh', 'linear']
        self.activation_function_last_dropdown.grid(row=7, column=1, sticky="ew", pady=5)

        # Dropdown-Menü zur Auswahl der Kostenfunktion (nur im Advanced-Modus sichtbar)
        self.cost_function_label = ttk.Label(self.main_frame, text="Kostenfunktion")
        self.cost_function_label.grid(row=8, column=0, sticky="w")
        self.cost_function_var = tk.StringVar()
        self.cost_function_var.set('categorical_crossentropy')
        self.cost_function_dropdown = ttk.Combobox(self.main_frame, textvariable=self.cost_function_var)
        self.cost_function_dropdown['values'] = ['mean_squared_error', 'mean_absolute_error', 'categorical_crossentropy', 'binary_crossentropy']
        self.cost_function_dropdown.grid(row=8, column=1, sticky="ew", pady=5)

        # Modellname (verschiebe nach unten um Platz für Kostenfunktion zu schaffen)
        ttk.Label(self.main_frame, text="Modellname:").grid(row=9, column=0, sticky="w")
        self.model_name_entry = ttk.Entry(self.main_frame)
        self.model_name_entry.grid(row=9, column=1, sticky="ew", pady=5)

        # Trainieren Button (verschiebe nach unten um Platz für Kostenfunktion zu schaffen)
        self.train_button = ttk.Button(self.main_frame, text="Training starten", command=self.start_training)
        self.train_button.grid(row=10, column=0, columnspan=2, pady=10, sticky="ew")

        # Modell speichern Button
        self.save_model_button = ttk.Button(self.main_frame, text="Modell speichern", command=self.save_model)
        self.save_model_button.grid(row=11, column=0, columnspan=2, pady=10, sticky="ew")

        self.toggle_mode()  # Initialisiert die Ansicht basierend auf dem ausgewählten Modus

    def toggle_mode(self, event=None):
        self.clear_graph()
        if self.mode.get() == "Beginner":
            self.activation_function_label.grid_remove()
            self.activation_function_dropdown.grid_remove()
            self.activation_function_last_label.grid_remove()
            self.activation_function_last_dropdown.grid_remove()
            self.cost_function_label.grid_remove()
            self.cost_function_dropdown.grid_remove()
        else:
            self.activation_function_label.grid()
            self.activation_function_dropdown.grid()
            self.activation_function_last_label.grid()
            self.activation_function_last_dropdown.grid()
            self.cost_function_label.grid()
            self.cost_function_dropdown.grid()

    def clear_graph(self):
        if self.canvas_labels:
            self.canvas_labels.get_tk_widget().grid_forget()
            self.canvas_labels = None
        if self.canvas:
            self.canvas.get_tk_widget().grid_forget()
            self.canvas = None

    def select_file(self):
        filepath = filedialog.askopenfilename()
        self.filepath_entry.delete(0, tk.END)
        self.filepath_entry.insert(0, filepath)

    def start_training(self):
        if self.training_in_progress:
            messagebox.showwarning("Warnung", "Ein Training läuft bereits.")
            return

        filepath = self.filepath_entry.get()
        if not filepath:
            messagebox.showwarning("Warnung", "Bitte wählen Sie eine Trainingsdatei aus.")
            return

        self.training_in_progress = True
        self.train_button.state(['disabled'])

        def train():
            test_data_percentage = float(self.test_data_percentage_entry.get()) / 100
            num_epochs = int(self.num_epochs_entry.get())
            hidden_layers = int(self.hidden_layers_var.get())
            hidden_units = int(self.hidden_units_var.get())
            activation = self.activation_function_var.get() if self.mode.get() == "Advanced" else 'relu'
            activation_last = self.activation_function_var_last.get() if self.mode.get() == "Advanced" else 'softmax'
            cost_function = self.cost_function_var.get() if self.mode.get() == "Advanced" else 'categorical_crossentropy'

            train_signals, train_labels, test_signals, test_labels = model_functions.prepare_data(filepath, test_data_percentage)
            self.model = model_functions.define_model((train_signals.shape[1],), train_labels.shape[1], hidden_layers, hidden_units, activation, activation_last)

            self.model.compile(optimizer='adam', loss=cost_function, metrics=['accuracy'])

            gui_update_callback = GUIUpdateCallback(self)
            self.model.fit(train_signals, train_labels, epochs=num_epochs, callbacks=[gui_update_callback])

            # Aktualisiere die Modellvisualisierung nach dem Training
            self.visualize_model_neurons()
            self.training_in_progress = False
            self.train_button.state(['!disabled'])

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
        if self.model is None:
            return

        plt.switch_backend('agg')  # Verwenden Sie das Agg-Backend für schnelleres Rendern

        fig_labels = Figure(figsize=(5, .5))
        ax_labels = fig_labels.add_subplot(111)
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)

        G = nx.DiGraph()

        # Initialisiere die Koordinaten für die Knotenplatzierung
        x = 0  # Startposition für die erste Schicht
        layer_width = 2  # Horizontale Distanz zwischen den Schichten

        pos = {}  # Dictionary zur Speicherung der Positionen der Knoten
        layers_neuron_indices = []  # Speichert die Indizes der Neuronen jeder Schicht
        node_colors = []  # Speichert die Farben für die Knoten

        # Eingabeschicht
        input_shape = self.model.input_shape[1]
        input_neurons = [f"Input_{i+1}" for i in range(input_shape)]
        layers_neuron_indices.append(input_neurons)
        input_color = '#1f78b4'  # Blau für die Eingabeschicht
        input_spacing = 2  # Beliebiges spacing für die Eingabeschicht
        y_offset_input = (input_shape - 1) * input_spacing / 2  # Zentriere die Neuronen vertikal in der Schicht
        for i, neuron in enumerate(input_neurons):
            pos[neuron] = (x, i * input_spacing - y_offset_input)  # Platzierung der Eingabeneuronen
            node_colors.append(input_color)  # Farbe für die Eingabeschicht

        x += layer_width

        # Versteckte Schichten und Ausgabeschicht
        for i, layer in enumerate(self.model.layers):
            layer_neurons = []
            num_neurons = layer.units if hasattr(layer, 'units') else 1  # Anzahl der Neuronen in der Schicht
            neuron_spacing = input_spacing * (input_shape / num_neurons)  # Berechne das neuron spacing basierend auf der Anzahl der Neuronen
            neuron_spacing = max(neuron_spacing, 0.5)  # Mindestabstand festlegen
            y_offset = (num_neurons - 1) * neuron_spacing / 2  # Zentriere die Neuronen vertikal in der Schicht

            for j in range(num_neurons):
                neuron_index = f"L{i+1}_N{j+1}"
                G.add_node(neuron_index)
                layer_neurons.append(neuron_index)

                # Setze die Position für jeden Neuronen-Knoten
                pos[neuron_index] = (x, j * neuron_spacing - y_offset)  # Gleichmäßig verteilte y-Werte
                node_colors.append(0)  # Dummy-Wert für die Eingabeschicht

            layers_neuron_indices.append(layer_neurons)
            x += layer_width  # Verschiebe die x-Position für die nächste Schicht

        # Verbinde die Neuronen zwischen den Schichten und füge die Gewichtungen als Kantenattribute hinzu
        edge_weights = []
        for i in range(len(layers_neuron_indices) - 1):
            weights, _ = self.model.layers[i].get_weights()

            if i == 0:
                for src in input_neurons:
                    for k, dst in enumerate(layers_neuron_indices[i + 1]):
                        weight = weights[:, k].mean()  # Durchschnittsgewicht aller Eingaben zu einem Neuron
                        G.add_edge(src, dst, weight=weight)
                        edge_weights.append(weight)
            else:
                for j, src in enumerate(layers_neuron_indices[i]):
                    for k, dst in enumerate(layers_neuron_indices[i + 1]):
                        weight = weights[j, k]
                        G.add_edge(src, dst, weight=weight)
                        edge_weights.append(weight)

        # Normiertes Farbschema für die Gewichtungen
        weight_norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights), clip=True)
        weight_cmap = plt.cm.RdYlGn  # Farbverlauf von rot (negativ) zu grün (positiv)

        # Zeichne den Graphen mit farbkodierten Kanten basierend auf den Gewichtungen
        edge_colors = [weight_cmap(weight_norm(data['weight'])) for u, v, data in G.edges(data=True)]

        nx.draw(G, pos, ax=ax, with_labels=False, node_size=100, node_color=input_color, font_size=8, font_weight="light", arrows=True, edge_color=edge_colors)

        # Schichtbeschriftungen hardcoden, "Eingabeschicht Versteckte Schichten Ausgabeschicht"
        ax_labels.text(.08, 0.5, 'Eingabeschicht', ha='center', va='center', fontsize=8, fontweight='bold', color='white', backgroundcolor=input_color)
        ax_labels.text(.46, 0.5, 'Versteckte Schichten', ha='center', va='center', fontsize=8, fontweight='bold', color='white', backgroundcolor=input_color)
        ax_labels.text(.84, 0.5, 'Ausgabeschicht', ha='center', va='center', fontsize=8, fontweight='bold', color='white', backgroundcolor=input_color)
        

        ax_labels.axis('off')

        # Legende für Gewichtungen
        sm = plt.cm.ScalarMappable(cmap=weight_cmap, norm=weight_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.08)
        cbar.set_label('Gewichte', rotation=270, labelpad=10)

        # Erstelle ein FigureCanvasTkAgg-Objekt und binde es an das Tkinter-Fenster
        canvas_labels = FigureCanvasTkAgg(fig_labels, master=self.main_frame)  # 'self.main_frame' ist das Tkinter-Hauptfenster
        canvas_labels.draw()
        canvas_labels.get_tk_widget().grid(row=12, column=0, columnspan=2, pady=0, sticky="nsew")

        canvas = FigureCanvasTkAgg(fig, master=self.main_frame)  # 'self.main_frame' ist das Tkinter-Hauptfenster
        canvas.draw()
        canvas.get_tk_widget().grid(row=13, column=0, columnspan=2, pady=10, sticky="nsew")

        # Konfiguriere das Grid im Haupt-Frame, um den Canvas aufzunehmen
        self.main_frame.grid_rowconfigure(12, weight=1)
        self.main_frame.grid_rowconfigure(13, weight=10)
        for col in range(2):
            self.main_frame.grid_columnconfigure(col, weight=1)

class GUIUpdateCallback(tf.keras.callbacks.Callback):
    def __init__(self, app):
        super().__init__()
        self.app = app

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        message = f"Epoch {epoch + 1}: Genauigkeit = {accuracy*100:.2f}%\n"
        self.app.root.after(0, self.app.update_console, message)

    def on_train_end(self, logs=None):
        self.app.root.after(0, self.app.update_console, "\nTraining abgeschlossen.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
