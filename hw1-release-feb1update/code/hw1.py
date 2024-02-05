import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt

def linear_gd(X, Y, lrate=0.1, num_iter=1000):
    # return parameters as numpy array
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    w = np.zeros(X.shape[1])
    n = len(Y)
    for i in range(num_iter):
        grad = 1 / n * np.dot(X.T, (np.dot(X, w) - Y))
        w = w - lrate * grad
    return w

def linear_normal(X, Y):
    # return parameters as numpy array
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)

def plot_linear():
    # return plot
    X, Y = utils.load_reg_data()
    
    w = linear_normal(X, Y)
    # w = linear_gd(X, Y)
    print(X.shape, Y.shape, w.shape)
    plt.scatter(X, Y, color='blue', label='Data Points')

    X_with_one = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    plt.plot(X, np.dot(X_with_one, w), color='red', label='Linear Function: y = {}x'.format(w))

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data Points and Linear Function')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_linear()