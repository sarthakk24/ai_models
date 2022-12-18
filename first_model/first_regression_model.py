import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import math

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# def compute_gradient_descent(w , b ) :

def compute_model_output(features , w , b ) :
    m = features.shape[0]
    
    # making an numpy array of size m default values as zeroes
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * features[i] + b

    return f_wb

def compute_cost_function(target , estimations):
    m = target.shape[0]
    val = 0
    for i in range(m):
        val += math.pow(estimations[i] - target[i] , 2) # type: ignore
    return 1/ (2 * m) * val

def compute_gradient(features , targets , w , b) :
    m = features.shape[0]
    dj_dw = 0
    dj_db = 0
    f_wb = compute_model_output(features , w, b)
    for i in range(m):
        dj_dw_i = (f_wb[i] - targets[i]) * features[i]
        dj_db_i = f_wb[i] - targets[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw , dj_db

def gradient_descent(features , target , w_initial , b_initial , alpha , num_iters ,cost_function , gradient_function):

    J_history = []
    p_history = []
    b = b_initial
    w = w_initial
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(features, target, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            estimations = compute_model_output(features , w , b)
            J_history.append( cost_function(features ,  estimations))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                f"w: {w: 0.3e}, b:{b: 0.5e}")
                
    return w, b, J_history, p_history


# Get the base path of the current working directory
base_path = os.getcwd()

# Join the base path with the relative path of the file
file_path = os.path.join(base_path , "first_model/student_scores.csv")

# used to read the csv file 
df = pd.read_csv(file_path)

# using pandas to read specific columns of the csv file
data_hours = df.loc[: , "Hours"].values
data_scores = df.loc[:, "Scores"].values

# storing the arrays in the form of numpy arrays
features = np.array(data_hours)
target = np.array(data_scores)

# setting the number of inputs used to train the model
m = target.shape[0]
print(f"Number of training examples is: {m}")


# using the plot library to plot the input and output graph 
plt.scatter(features, target, marker='*', c='r')  # type: ignore
plt.title("Student scores vs hours spent")
plt.ylabel('Hours spent')
plt.xlabel('Scores')
plt.show()

# setting the values of weights in the model 
w = 0
b = 0
tmp_alpha = 1.0e-2
iterations = 1000

w_final, b_final, J_hist, p_hist = gradient_descent(features ,target, w, b, tmp_alpha, iterations, compute_cost_function, compute_gradient)

estimations = compute_model_output(features , w_final , b_final)
j_wb = compute_cost_function(target , estimations)

# plotting our predictions line 
plt.plot(features, estimations, c='b',label=f"Our Prediction j_wb : {j_wb}")
plt.scatter(features,target, marker='*', c='r',label='Actual Values') # type: ignore
plt.title("Estimation")
plt.ylabel('Hours spent')
plt.xlabel('Scores')
plt.legend()
plt.show()
# print(w_final , b_final )

