import numpy as np

def backward(self, dout):
    self.db = np.sum(dout, axis=0)
    self.dW = self.X.T.dot(dout)
    dX = dout.dot(self.W.T)