import sys
import os
import warnings
import math
import numpy as np
import matplotlib.pyplot as plt

# Suppress endless warnings from tensorflow.
# warnings.filterwarnings('ignore')

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Load specified dataset.
input_file = './data/pima-indians-diabetes-training.data.csv'
if not os.path.exists(input_file):
	print('Data file', input_file, 'not found.')
	exit(1)
dataset = loadtxt(input_file, delimiter=',')

# Split up training data into independent and dependent variables.
x = dataset[:,0:8]
y = dataset[:,8]

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

# The third argument which contains unseen data.
unseen_file = './data/pima-indians-diabetes-testing.data.csv'
if not os.path.exists(unseen_file):
	print('Unseen file', unseen_file, 'not found.')
	exit(1)

# Load and predict.
unseen = loadtxt(unseen_file, delimiter=',')
results = model.predict(unseen[:,0:8])

# Compile deciles.
counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for result in results:
	counts[math.floor(result[0] * 10)] += 1

# Plot bar graph.
plt.bar(['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100'], counts)
plt.title('Predicted chances of having diabetes in unseen population')
plt.xlabel('Chance of having diabetes (%)')
plt.ylabel('Population count')
plt.show()
