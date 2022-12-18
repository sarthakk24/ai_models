import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This imports the necessary libraries for this script. numpy is a library for working with arrays and mathematical operations on them. pandas is a library for working with data in a tabular format. matplotlib is a library for creating plots and charts.

def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

#This defines a function compute_model_output that takes in three arguments: x, w, and b. x is an array of input values, w is the weight or slope of the line, and b is the bias or y-intercept of the line. The function returns an array of output values, f_wb, which is the result of applying the linear model to the input values x.

x_train = np.array([1.0,2.0])
y_train = np.array([300.0,500.0])

#These lines create two numpy arrays, x_train and y_train, which contain the input and output training data, respectively.

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

i = 1 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

w = 200
b = 100
print(f"w: {w}")
print(f"b: {b}")

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
 
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")