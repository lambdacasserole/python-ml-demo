import sys
import os
import warnings
import math
import numpy as np

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

def read_file(path):
	""" Reads the entire text of a file and returns it.
	Args:
		path (str): The path of the file to read from.
	Returns:
		str: The file contents.
	"""
	buffer = ''
	with open(path) as file:
		for line in file:
			buffer += line
	return buffer

# Check argument list length.
if len(sys.argv) < 3:
	print('Bad arguments.')
	exit(1)

# Load specified dataset.
input_file = sys.argv[-2]
if not os.path.exists(input_file):
	print('Data file', input_file, 'not found.')
	exit(1)
dataset = loadtxt(input_file, delimiter=',')

# Load labels for UX.
labels_file = sys.argv[-1]
if not os.path.exists(labels_file):
	print('Labels file', labels_file, 'not found')
	exit(1)
labels = [label.strip() for label in read_file(labels_file).split(',')]
input_cols = len(labels)

# Split up training data into independent and dependent variables.
x = dataset[:,0:input_cols]
y = dataset[:,input_cols]

# Build, compile and fit model.
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=10)

# Evaluate accuracy.
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy * 100))

# Read values in for prediction if in interactive mode.
next = True
while next:
	# Collect values according to labels.
	inputs = []
	for label in labels:
		value = input(f'Please enter ' + label.lower() + ': ')
		inputs.append(float(value))
	# Make prediction.
	result = model.predict(np.array(inputs, ndmin=2))
	print('Prediction: %.2f%%' % (result[0,0] * 100))
	next = input('Analyse next datapoint? [y/N]: ').lower()[0] == 'y' # Again?
