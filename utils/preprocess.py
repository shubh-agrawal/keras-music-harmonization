import os
import glob
import numpy as np
import midi_manipulation

data_path = os.path.join(".", "data")

def list_midis():
	midi_files = glob.glob('{}/*.*mid*'.format(data_path))
	return midi_files

def getStateMatrices():
	files = list_midis()
	stateMatrices = []
	i = 1
	for f in files:
		matrix = np.array(midi_manipulation.midiToNoteStateMatrix(f))
		if matrix.shape[0]>75:
			stateMatrices.append(matrix)
		os.system('clear')
		print "Loaded %d/%d files..."%(i, len(files))
		i = i+1
	return stateMatrices

def formatData(data, iLength, oLength, test_ratio = 0.15):
	# calculate the number of examples
	num = 0
	for d in data:
		num = num + (d.shape[0]-iLength-oLength)
	
	# format entire data
	X = np.zeros((num, iLength, 156))
	Y = np.zeros((num, oLength, 156))
	count = 0
	for d in data:
		for i in range(d.shape[0]-iLength-oLength):
			X[count][0:iLength] = d[i:i+iLength]
			Y[count][0:oLength] = d[i+iLength:i+oLength+iLength]
			count = count+1

	print "Database created."
	
	# shuffle the data
	p = np.random.permutation(len(X))
	X = X[p]
	Y = Y[p]

	print "Database shuffled."

	# test-train split
	x_train = X[0:int((1-test_ratio)*len(X))]
	y_train = Y[0:int((1-test_ratio)*len(Y))]
	print "Created training data."

	x_test = X[int((1-test_ratio)*len(X)):len(X)]
	y_test = Y[int((1-test_ratio)*len(Y)):len(Y)]
	print "Created test data."

	return x_train, y_train, x_test, y_test

def loadData(iLength=5, oLength=5):
	# check if foramtted data is present
	if os.path.exists("dataset"):
		print "Formatted data found..."

		x_train = np.load("dataset/x_train.npy")
		y_train = np.load("dataset/y_train.npy")
		print "Loaded training data."

		x_test = np.load("dataset/x_test.npy")
		y_test = np.load("dataset/y_test.npy")
		print "Loaded test data."
	else:
		os.mkdir("dataset")
		
		# check if raw data is present
		if os.path.exists("state_matrices.pkl"):
			print "Loading from serialized list..."
			data = pickle.load(open("state_matrices.pkl", 'rb'))
		else:
			print "Loading from midis..."
			data = load_data.getStateMatrices()
			pickle.dump(data, open("state_matrices.pkl", 'wb'))

		x_train, y_train, x_test, y_test = load_data.formatData(data, iLength, oLength)

		np.save("dataset/x_train.npy", x_train)
		np.save("dataset/y_train.npy", y_train)
		print "Training data saved."

		np.save("dataset/x_test.npy", x_test)
		np.save("dataset/y_test.npy", y_test)
		print "Test data saved."

	# return the sets
	return x_train, y_train, x_test, y_test