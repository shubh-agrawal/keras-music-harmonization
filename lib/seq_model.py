import numpy as np
import seq2seq
from seq2seq.models import Seq2Seq
import keras.optimizers as optim 

# simple Seq2Seq model: with t:t+x as input, t+x:t+x+y as output
# expected shape for model.fit(x,y): x - num x iLength x 156, x - num x oLength x 156
def createModel(iLength=5, oLength=5, hidden=32, d=1, learning_rate=0.001):
	model = Seq2Seq(input_dim=156, input_length=iLength, hidden_dim=hidden, output_dim=156, output_length=oLength, depth=d)
	opt = optim.RMSprop(lr=learning_rate)
	model.compile(loss='mse', optimizer=opt)
	return model