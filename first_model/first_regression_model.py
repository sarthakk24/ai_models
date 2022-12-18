import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import math

def compute_model_output(features , w , b ) :
    m = features.shape[0]
    
    # making an numpy array of size m default values as zeroes
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * features[i] + b

    return f_wb

def compute_cost_function(target , estimations):
    m = features.shape[0]
    val = 0
    for i in range(m):
        val += math.pow(estimations[i] - target[i] , 2)
    return val / (2 * m)
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
w = 10
b = 1.5

estimations = compute_model_output(features , w , b)
j_wb = compute_cost_function(target , estimations)
# plotting our predictions line 
plt.plot(features, estimations, c='b',label=f"Our Prediction j_wb : {j_wb}")
plt.scatter(features,target, marker='*', c='r',label='Actual Values') # type: ignore
plt.title("Estimation")
plt.ylabel('Hours spent')
plt.xlabel('Scores')
plt.legend()
plt.show()

print(m)

