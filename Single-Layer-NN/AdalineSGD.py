import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Perceptron import plot_decision_regions

class AdalineSGD():
  """ADAptive LInear NEuron classifier.
  
  Parameters
  ____________
  eta : float
    Learning rate (between 0.0 and 1.0)
  n_iter : int
    Passes oover the training dataset.
  shuffle : bool (default: True)
    Shuffles training data every epoch if True to prevent cycles.
  random_state : int
    Random number generator seed for random weight initialization.

  Attributes
  ____________
  w_ : 1d-array
    Weights after fitting.
  cost_ : list
    Sum-of-squares cost function value averaged over all training examples in each epoch
  """
  def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
    self.eta = eta
    self.n_iter = n_iter
    self.shuffle = shuffle
    self.random_state = random_state

  def fit(self, X, y):
    """ Fit training data.
    
    Parameters
    ____________
    X : {array-like}, shape = [n_examples. n_features]
      Training vectors, where n_examples is the number of examples and n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    """
    self._initialize_weights(X.shape[1])
    self.cost_ = []
    for i in range(self.n_iter):
      if self.shuffle:
        X, y = self._shuffle(X, y)
      cost = []
      for xi, target in zip(X, y):
        cost.append(self._update_weights(xi, target))
      avg_cost = sum(cost) / len(y)
      self.cost_.append(avg_cost)
    return self

  def partial_fit(self, X, y):
    """Fit training data without reinitializing the weights"""
    if not self.w_initialized:
      self._initialize_weights(X.shape[1])
    if y.ravel.shape[0] > 1:
      for xi, target in zip(X, y):
        self._update_weights(xi, target)
    else:
      self._update_weight(X, y)
    return self

  def _shuffle(self, X, y):
    """Shuffle training data"""
    r = self.rgen.permutation(len(y))
    return X[r], y[r]

  def _initialize_weights(self, m):
    """Initialize weights to small random numbers"""
    self.rgen = np.random.RandomState(self.random_state)
    self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
    self.w_initialized=True

  def _update_weights(self, xi, target):
    """Apply Adaline learning rule to update the weights"""
    output = self.activation(self.net_input(xi))
    error = (target - output)
    self.w_[1:] += self.eta * xi.dot(error)
    self.w_[0] += self.eta * error
    cost = 0.5 * error**2
    return cost

  def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_[1:]+self.w_[0])

  def activation(self, X):
    """Compute linear activation"""
    return X

  def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)      
  

s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
print('URL', s)
df = pd.read_csv(s, header=None, encoding='utf-8')
y = df.iloc[0:100, 4].values                # select the first 100 class labels in the Iris dataset, which contain the measurements for Setosa and Versicolor
y = np.where(y == 'Iris-setosa', 1, -1)     # replace each class label with it's corresponding integer: 1 for setosa and -1 for versicolor
X = df.iloc[0:100, [0, 2]].values           # extract the sepal and petal lengths
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_sgd.cost_) + 1), ada_sgd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.show()
