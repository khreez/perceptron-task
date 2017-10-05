# inputs assumes a bias added with a value of 1 as the last elemnt of the array
def activation(inputs, weights):
  activated = 0.0
  for i in range(len(inputs)):
    # weighted sum of inputs, including bias
    activated += inputs[i] * weights[i]
  return activated

def prediction(inputs, weights):
  activated = activation(inputs, weights)
  # binary threshhold classifier
  return 1.0 if activated >= 0.0 else 0.0

# train_set needs to include a bias and an expected output value
# as the last two elements of each input row
def train_weights(train_set, learning_rate):  
  # weights initialization, could be a random from -0.5 to 0.5 skipping 0
  weights = [0.0 for i in range(len(train_set[0]) - 1)]
  for row in train_set:
    # discard the expected output value (last array item)
    inputs = row[:-1]
    predicted = prediction(inputs, weights)
    error = row[-1] - predicted
    # do we need to train wheights until prediction matches all expected classes?
    for i in range(len(weights)):
      # stochastic gradient descent weigth adjustment
      weights[i] = weights[i] + learning_rate * error * inputs[i]
  return weights

data_set = [
  [2.7810836,2.550537003,0],
  [1.465489372,2.362125076,0],
  [3.396561688,4.400293529,0],
  [1.38807019,1.850220317,0],
  [3.06407232,3.005305973,0],
  [7.627531214,2.759262235,1],
  [5.332441248,2.088626775,1],
  [6.922596716,1.77106367,1],
  [8.675418651,-0.242068655,1],
  [7.673756466,3.508563011,1]
]

# new array to accomodate bias from a 60% data_set trainig sample
train_set = [[row[0], row[1], 1, row[2]] for row in data_set[:6]]
learning_rate = 0.9

# first train the weights
weights = train_weights(train_set, learning_rate)
print(weights)

for row in data_set:
  # adding bias to the input
  inputs = [row[0], row[1], 1]
  # execution
  predicted = prediction(inputs, weights)
  print("Expected=%d, Predicted=%d" % (row[-1], predicted))
