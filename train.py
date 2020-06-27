import numpy as np 
import os
import glob
from utils import preprocess
from lib import seq_model
import pickle

x_train, y_train, x_test, y_test = preprocess.loadData()

model = seq_model.createModel()