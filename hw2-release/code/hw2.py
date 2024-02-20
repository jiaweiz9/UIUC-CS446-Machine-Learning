import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    # x_train= torch.cat((torch.ones((x_train.shape[0], 1)), x_train), axis=1)
    kernel_matrix = torch.zeros(x_train.size(0), x_train.size(0))
    for i in range(x_train.size(0)):
        for j in range(x_train.size(0)):
            kernel_matrix[i, j] = kernel(x_train[i], x_train[j]) * y_train[i] * y_train[j]

    alpha = torch.zeros(x_train.size(0), requires_grad=True)
    print(alpha)
    optimizer = optim.SGD([alpha], lr=lr)
    
    for i in range(num_iters):
        mini_problem = -torch.sum(alpha) + 0.5 * alpha @ kernel_matrix @ alpha
        mini_problem.backward()
        optimizer.step()
        with torch.no_grad():
            alpha.clamp_(min=0, max=c)
        alpha.requires_grad_()
        optimizer.zero_grad()
    
    return alpha.detach()


def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    # x_train= torch.cat((torch.ones((x_train.shape[0], 1)), x_train), axis=1)
    # x_test= torch.cat((torch.ones((x_test.shape[0], 1)), x_test), axis=1)
    predict = torch.zeros(x_test.size(0))
    for j in range(x_test.size(0)):
        for i in range(x_train.size(0)):
            predict[j] += alpha[i] * y_train[i] * kernel(x_train[i], x_test[j])
    return predict

if __name__ == '__main__':
    x, y = hw2_utils.xor_data()
    alpha = svm_solver(x, y, lr=0.01, num_iters=1000, kernel=hw2_utils.rbf(sigma=4))
    # print(alpha)
    # print(svm_predictor(alpha, x, y, x))

    hw2_utils.svm_contour(lambda x_test: svm_predictor(alpha, x, y, x_test, kernel=hw2_utils.rbf(sigma=4)))
    # print('The accuracy of the SVM is: ', torch.mean((svm_predictor(alpha, x, y, x) == y).float()).item())
    # plt.show()