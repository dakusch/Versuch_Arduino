"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self, threshold=0.5, windowsize=10):  # Schwellwert als Parameter
        gr.sync_block.__init__(
            self,
            name='Moving Average with Threshold Decision',
            in_sig=[np.float32],  # Verwendung von float32 für Eingangssignal
            out_sig=[np.int32]    # Ausgangssignal als int32 (für 0 oder 1 Werte)
        )
        self.threshold = threshold
        self.window_size = windowsize
        self.buffer = np.zeros(self.window_size, dtype=np.float32)  # Buffer als float32

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        
        for i in range(len(in0)):
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = in0[i]
            moving_average = np.mean(self.buffer)
            out[i] = 1 if moving_average > self.threshold else 0
            
        print(len(out))
            
        return len(out)
