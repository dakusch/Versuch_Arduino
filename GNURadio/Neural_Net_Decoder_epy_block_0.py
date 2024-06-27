import numpy as np
from gnuradio import gr
from tensorflow.keras.models import load_model

class KerasOOKDecoder(gr.sync_block):
    def __init__(self, offset=0):
        gr.sync_block.__init__(self,
                               name="Keras OOK Decoder",
                               in_sig=[np.float32],
                               out_sig=[np.int32])
        self.model = load_model('/home/david/Nextcloud/Arbeit_ANT/Versuch_Arduino/Neural_Net/data/tes_new.keras')
        self.symbol_length = 100
        self.offset = offset  # Offset-Parameter hinzugefügt

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        
        # Prüfen, ob genug Daten vorhanden sind, um die Verarbeitung zu starten
        if len(in0) < self.symbol_length + self.offset:
            return 0  # Nicht genug Daten, um Offset und ein vollständiges Symbol zu behandeln

        # Beginnend vom Offset die Daten für das Modell vorbereiten
        start_index = self.offset
        end_index = start_index + (len(in0) - self.offset) // self.symbol_length * self.symbol_length
        reshaped_samples = np.reshape(in0[start_index:end_index], (-1, self.symbol_length))

        # Das Modell auf alle Samples gleichzeitig anwenden
        predictions = self.model.predict(reshaped_samples).round().astype(np.int32).flatten()

        # Die Vorhersagen in die Ausgabe kopieren
        out[:len(predictions)] = predictions
        
        return len(predictions)
