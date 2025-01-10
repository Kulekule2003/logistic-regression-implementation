import numpy as  np
import matplotlib.pyplot as plt
import pandas as pd
import math

titanic_data = pd.read_csv("data/train.csv")

#decoding the sex
titanic_data['Sex'] = titanic_data['Sex'].map({'male':0,'female':1})

#creating numpy arrays for the x and y training sets
#using only the required fields for x {sex, parch, fare}

requ_x = ['Sex','Parch','Fare']
requ_y = ['Survived']

required_data_x = titanic_data[requ_x]
required_data_y = titanic_data[requ_y]

#converting the the required data into numpy arrays
# we need to deal with sex , sibsp, fare
X_train = required_data_x.to_numpy()
y_train = required_data_y.to_numpy()

#defining the sigmoid function
def sigmoid(z):
    g_z = 1/(1+np.exp(-z))
    return g_z   

# the compute cost 
def compute_cost(X,y,w,b):
    """ 
    w - is an array of parameters
    b - is one parameter
    X - is an array of training inputs
    y - is an array of training outputs
    """
    m = X.shape[0]
    cost = 0
    for i in range(m):
        z = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost= cost/(2*m)
    return cost

# computing the gradient   
def compute_gradient(X,y,w,b):
    """
    m - number of inputs
    n - the number of features
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        z = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z)
        error = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + error * X[i][j]
        dj_db = dj_db + error

    dj_dw = dj_dw/m
    dj_db = dj_db/m
    
    return dj_dw,dj_db 

# the gradient descent
def gradient_descent(X,y,w_in,b_in,num_iters,alpha):
    """ num_iters - the number of iterations 
    w_in - an array of parameters
    b_in - an array of parameters
    """
    J_history = []

    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X,y,w,b)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        J_history.append(compute_cost(X,y,w,b))
            
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}  ,w :{w}, b : {b} ")
    
    return w, b, J_history
    
n = X_train.shape[1]
w_in = np.zeros((n))
b_in = 0
w_out,b_out, history= gradient_descent(X_train,y_train,w_in,b_in,100000,0.001)

#predicting by entering another input
def test_algo(X,w_out,b_out):
    f_wb = np.dot(X,w) + b
    return f_wb