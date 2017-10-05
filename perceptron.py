from csv import reader
from random import sample

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
  attempt = 0
  is_trained = False
  while is_trained is False:
    attempt += 1
    sum_error = 0.0
    is_matching = True
    for row in train_set:
      # discard the expected output value (last array item)
      inputs = row[:-1]
      predicted = prediction(inputs, weights)
      error = row[-1] - predicted
      # do we need to train wheights until prediction matches all expected classes?
      is_matching = is_matching and bool(error == 0)
      sum_error += error**2
      for i in range(len(weights)):
        # stochastic gradient descent weigth adjustment
        weights[i] = weights[i] + learning_rate * error * inputs[i]
    is_trained = is_matching
    print('>attempt=%d, lrate=%.3f, error=%.3f' % (attempt, learning_rate, sum_error))
  return weights

def select_train_set(data_set):
  sample_size = int(len(data_set) * 0.6)
  # print("size: %d, sample: %d" % (len(data_set), sample_size))
  # shuffling data
  shuffled_set = sample(data_set, len(data_set))
  # new array to accomodate bias from a 60% data_set trainig sample
  train_set = [[row[0], row[1], 1, row[2]] for row in shuffled_set[:sample_size]]
  return train_set

def perceptron(data_set, learning_rate):
  # get a training set
  train_set = select_train_set(data_set)
  # first train the weights
  weights = train_weights(train_set, learning_rate)
  # print(weights)
  predictions = list()
  for row in data_set:
    # adding bias to the input
    inputs = [row[0], row[1], 1]
    # execution
    predicted = prediction(inputs, weights)
    predictions.append(predicted)
    print("Expected=%d, Predicted=%d" % (row[-1], predicted))
  return predictions

#===== BEGIN | utility functions

# Convert string column to integer
def str_column_to_int(values, column):
  class_values = [row[column] for row in values]
  unique = set(class_values)
  lookup = dict()
  for i, value in enumerate(unique):
    lookup[value] = i
  for row in values:
    row[column] = lookup[row[column]]
  return lookup

# Load a CSV file
def load_csv(file_name):
  inputs = list()
  with open(file_name, 'r') as file:
    csv_reader = reader(file)
    for row in csv_reader:
      if not row:
        continue
      # print(row)
      inputs.append([float(row[0].strip()), float(row[1].strip()), row[-1]])
  return inputs

# Load a CSV file and transform it to a dataset
def load_data_set(file_name):
  data_set = load_csv(file_name)
  str_column_to_int(data_set, len(data_set[0])-1)
  return data_set

#===== END | utility functions

data_set = load_data_set('iris.data')
# merge unwanted class values for iris.data
for row in data_set:
  row[-1] = 1 if row[-1] == 2 else 0
  # print(row)

learning_rate = 0.5
perceptron(data_set, learning_rate)
