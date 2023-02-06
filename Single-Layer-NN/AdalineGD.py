import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Perceptron import plot_decision_regions

class AdalineGD(object):
  """ADAptive LInear NEyron classifier.

  Parameters
  _____________
  eta : float
    Learning rate (between 0.0 and 1.0)
  n_iter : int
    Passes over the training dataset (epoch).
  random_state : int
    Random number generator seed for random weight initialization

  * Perceptron Hyperparameters:
    The learning rate (eta) and the epochs (n_iter) are the hyperparameters (or tunint parameters)
    of the Perceptron and Adaline learning algorithmns.
  
  Attributes
  ____________
  w_ : 1d-array
    Weights after fitting.
  cost_ : list
    Sum-of-squares cost function value in each epoch
  """
  def __init__(self, eta=0.1, n_iter=50, random_state=1):
    self.eta = eta
    self.n_iter = n_iter
    self.random_state = random_state

  def fit(self, X, y):
    """ Fit training data.
    
    Parameters
    ___________
    X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples
        is the number of examples and 
        n_features is the number of features.
    y : array-like, shape= [n_examples]
        Target values.

    Returns
    _______
    self : object
    """
    rgen = np.random.RandomState(self.random_state)
    self.w_ = rgen.normal(loc=0.0, scale=0.1,
                          size=1+X.shape[1])
    self.cost_ = []

    for i in range(self.n_iter):
      net_input = self.net_input(X)
      output = self.activation(net_input)
      errors = (y - output)
      self.w_[1:] += self.eta * X.T.dot(errors)   
      self.w_[0] = self.eta * errors.sum()          # calculate the gradient
      cost = (errors**2).sum() / 2.0                
      self.cost_.append(cost)
    return self

  def net_input(self, X):
    """Calculate net input"""
    return np.dot(X, self.w_[1:]) + self.w_[0]
  

  # Has no effect since it is an identity function.
  # Added here to illustrate the general concept
  # with regard to how information flows through a single-
  # layer neural network:
  # Features from the input data, net input, activation and output
  def activation(self, X):  
    """Compute linear activation"""
    return X
  
  def predict(self, X):
    """Return class label after unit step"""
    return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
  

s = os.path.join('https://archive.ics.uci.edu', 'ml', 'machine-learning-databases', 'iris', 'iris.data')
print('URL', s)
df = pd.read_csv(s, header=None, encoding='utf-8')
df.tail()

y = df.iloc[0:100, 4].values                # select the first 100 class labels in the Iris dataset, which contain the measurements for Setosa and Versicolor
y = np.where(y == 'Iris-setosa', 1, -1)     # replace each class label with it's corresponding integer: 1 for setosa and -1 for versicolor

X = df.iloc[0:100, [0, 2]].values           # extract the sepal and petal lengths

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# The learning rate is too large. Instead of minimizing the cost function,
# the error becomes larger in every epoch, because we overshoot the global minimum.
ada1 = AdalineGD(n_iter=10, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

# The learning rate is so small that the algorithm would require a very 
# large number of epochs to converge to the global minimum
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("log(Sum-squared-error)")
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
print(f'DEBUG >>> X: {X}\n\tX_std: {X_std}')

ada_gd = AdalineGD(eta=0.01, n_iter=15)
ada_gd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
plt.show()