"""
A python script to get flow resistivity for fibrous porous acoustic model
from frequency-absorption coefficient curve obtained from impedance tube test

AUTHOR: Youngbin LIM
CONTACT: Youngbin.LIM@3ds.com

Description on the input parameter:

Test_Data_Path : Path to test data. Text file should be in f(Hz) vs Absorption coeff without column name (Numbers only). Delimiter can be either comma or tab
Porous_Model   : Set this value to 0 for Delany&Bazley, and 1 for Miki model
d              : Thickness of Porous material used in impedance tube test (in m)
r_min, r_max   : Range of Flow resistivity for searching
obj            : Error calculation method. Set this value to 0 for average of absolute error, 1 for sum of error square
c0             : Speed of sound in air in m/s
rho0           : Density of air in m/kg^3

"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

################################
## Input parameter definition ##
################################

Test_Data_Path = "Test_Data.txt"
Porous_Model = 0
d = 0.025
r_min, r_max = 100.0, 1000000.0
obj = 0
c0 = 343.0
rho0 = 1.2

#########################
## Calculate constants ##
#########################

if Porous_Model==0: # Delany & Bazley 
    C1 = 0.0978; C2 = -0.7; C3 = 0.189; C4 = -0.595 
    C5 = 0.0571; C6 = -0.754; C7 = 0.087; C8 = -0.732;
elif Porous_Model==1: # Miki 
    C1 = 0.1227; C2 = -0.618; C3 = 0.1792; C4 = -0.618; 
    C5 = 0.0786; C6 = -0.632; C7 = 0.1205; C8 = -0.632;
    

def guess_delimiter_and_load(filename):
    # Read the first line of the file
    with open(filename, 'r') as f:
        first_line = f.readline()
    
    # Guess the delimiter based on the first line
    tab_count = first_line.count('\t')
    comma_count = first_line.count(',')

    if tab_count > comma_count:
        delimiter = '\t'
    else:
        delimiter = ','

    # Use the determined delimiter with genfromtxt
    return np.genfromtxt(filename, delimiter=delimiter)

TestData = guess_delimiter_and_load(Test_Data_Path)

# Assign frequency and absorption coefficient array
freq = TestData[:,0]
alpha_test = TestData[:,1]

# Error function for a given value of r
def error_function(r):
    X = (rho0*freq)/r
    k0 = 2*np.pi*freq/c0
    k = k0*(1.0 + C1*np.power(X, C2)) + k0*C3*np.power(X, C4)*-1j
    Zc = (1.0 + C5*np.power(X, C6)) + (C7*np.power(X, C8)) * -1j
    
    Zs = -1j*Zc*1/np.tan(k*d)
    alpha_pred = 1.0 - np.power(abs((Zs-1)/(Zs+1)),2)
    
    Err_arr = 100*abs((alpha_test-alpha_pred)/alpha_test)
    if obj==0:
        obj_fun=np.average(np.abs(Err_arr))
    elif obj==1:
        obj_fun=np.sum(np.power(Err_arr,2))
    return obj_fun

# Generate Random samples for r
n = 1000
r_samples = r_min + (r_max - r_min) * np.random.rand(n)

# Evaluate the error for each sample
errors = [error_function(r) for r in r_samples]

# Choose the best r value (one with the minimum error)
best_r_index = np.argmin(errors)
best_r_init = r_samples[best_r_index]

# Use this best initial value of r for optimization
result = minimize(error_function, best_r_init, method = 'SLSQP', bounds=[(r_min, r_max)], 
                  options={'maxiter': 100000, 'ftol': 1e-9, 'disp': True})

r_optimized = result.x[0]

# Function to compute alpha_pred for a given value of r
def compute_alpha_pred(r):
    X = (rho0*freq)/r
    k0 = 2*np.pi*freq/c0
    k = k0*(1.0 + C1*np.power(X, C2)) + k0*C3*np.power(X, C4)*-1j
    Zc = (1.0 + C5*np.power(X, C6)) + (C7*np.power(X, C8)) * -1j
    
    Zs = -1j*Zc*1/np.tan(k*d)
    return 1.0 - np.power(abs((Zs-1)/(Zs+1)),2)

# Using the optimized value of r to get predicted alpha
alpha_pred = compute_alpha_pred(r_optimized)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(freq, alpha_test, label='Test Data', marker='o', markersize=4, linestyle='-')
plt.plot(freq, alpha_pred, label='Predicted', marker='x', markersize=4, linestyle='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Absorption Coefficient')
plt.legend()
plt.grid(True)

# Add optimized value as text
text_str = "Optimized value of flow resistivity is: {:.2f} (Pa*s/m^2)".format(r_optimized)
plt.text(0.1 * max(freq), 0.1, text_str, 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

plt.show()
