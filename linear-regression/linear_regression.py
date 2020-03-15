import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

test_x = np.array([2, 4, 5, 6, 9, 11, 15, 17, 22, 24, 26, 28, 30])
test_y = np.array([5, 7, 6, 9, 11, 10, 12, 13, 2, 15, 1, 17, 18])

def get_linear_func(params):
  a = params[0]
  b = params[1]
  return lambda x: a * x + b

def draw_subplot(func, grid, title):
  plt.subplot(grid)
  plt.axis([0, max(test_x) + 1, 0, max(test_y) + 1])
  plt.plot(test_x, test_y, 'go')
  plt.plot([0, test_x[-1]], [func(0), func(test_x[-1])])
  plt.title(title)

# Calculating regression line using analytical methods
def calc_optimal_a(x, y):
  return (x * y).sum() / (x ** 2).sum()

def calc_optimal_a_and_b(x, y):
  n = x.size
  a = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x ** 2).sum() - (x.sum() ** 2))
  b = (y.sum() - a * x.sum()) / n
  return (a, b)


plt.suptitle('Linear regression')

primitive_regression_analytical = get_linear_func((calc_optimal_a(test_x, test_y), 0))
draw_subplot(primitive_regression_analytical, 221, 'Sum of squares, b = 0 (analytical)')

regression_analytical = get_linear_func(calc_optimal_a_and_b(test_x, test_y))
draw_subplot(regression_analytical, 222, 'Sum of squares (analytical)')


# Calculating regression line using numerical methods
def one_var_q(a):
  res = 0
  for i in range(0, test_x.size):
    res += abs(test_y[i] - (a[0] * test_x[i]))
  return res

one_var_q_min = minimize(one_var_q, [1]).x
primitive_regression_numerical = get_linear_func((one_var_q_min[0], 0))
draw_subplot(primitive_regression_analytical, 223, 'Absolute value, b = 0 (numerical)')

def two_var_q(a):
  res = 0
  for i in range(0, test_x.size):
    res += abs(test_y[i] - (a[0] * test_x[i] + a[1]))
  return res

two_var_q_min = minimize(two_var_q, [1, 1]).x
regression_numerical = get_linear_func((two_var_q_min[0], two_var_q_min[1]))
draw_subplot(regression_numerical, 224, 'Absolute value (numerical)')

plt.show()