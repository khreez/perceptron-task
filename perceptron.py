def activation(inputs, weights):
  activated = 0.0
  for i in range(len(inputs)):
    activated += inputs[i] * weights[i]
  return activated

def prediction(inputs, weights):
  activated = activation(inputs, weights)
  return 1.0 if activated >= 0.0 else 0.0

def train_weights(train_set, learning_rate):
  # weights = [0.0 for i in range(len(train_set[0]) - 1)]
  weights = [0.5, -0.1, .02]
  for row in train_set:
    inputs = row[:-1]
    predicted = prediction(inputs, weights)
    error = row[-1] - predicted
    for i in range(len(weights)):
      weights[i] = weights[i] + learning_rate * error * inputs[i]
  return weights

# data_set = [[2.7810836,2.550537003,0],
#   [1.465489372,2.362125076,0],
#   [3.396561688,4.400293529,0],
#   [1.38807019,1.850220317,0],
#   [3.06407232,3.005305973,0],
#   [7.627531214,2.759262235,1],
#   [5.332441248,2.088626775,1],
#   [6.922596716,1.77106367,1],
#   [8.675418651,-0.242068655,1],
#   [7.673756466,3.508563011,1]]
# weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
data_set =[
  [1.5,3,0],
  [2,1,0],
  [1,2,0],
  [3.5,4,1],
  [3,3,1],
  [4,2,1]
]
learning_rate = 0.9
train_set = [[row[0], row[1], 1, row[2]] for row in data_set]
train_set = [
  train_set[0],
  train_set[3],
  train_set[2],
  train_set[5]
]
print(train_set[:5])
weights = train_weights(train_set[:5], learning_rate)
print(weights)

for row in data_set:
  inputs = [row[0], row[1], 1]
  predicted = prediction(inputs, weights)
  print("Expected=%d, Predicted=%d" % (row[-1], predicted))
