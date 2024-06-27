import numpy as np
from gnuradio import gr

class OOK_Synchronizer(gr.sync_block):
    def __init__(self, threshold=0.5, symbol_length=100):
        gr.sync_block.__init__(self,
                               name="OOK Synchronizer",
                               in_sig=[np.float32],
                               out_sig=[np.float32])
        self.threshold = threshold
        self.symbol_length = symbol_length
        self.synced = False
        self.index = 0

    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        noutput_items = len(in0)

        for i in range(noutput_items):
            if not self.synced:
                if in0[i] > self.threshold:
                    self.synced = True
                    self.index = 0
            if self.synced:
                if self.index < self.symbol_length:
                    out[self.index] = in0[i]
                    self.index += 1
                if self.index == self.symbol_length:
                    self.synced = False
                    self.index = 0

        return self.index
