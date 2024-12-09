import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from BFV import *  # Import the BFV module

# Load the dataset
file = 'kc_house_data.csv'
df = pd.read_csv(file)

# Check if the dataframe is not empty
if not df.empty:
    print('Dataframe imported successfully!')

# Preprocess the data
df.drop(columns=['id', 'date'], inplace=True)  # Drop 'id' and 'date' columns
df.dropna(inplace=True)  # Drop rows with missing values

# Define features and target variable
X = df[['sqft_living', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront']]  # Select a few features for testing
y = df['price']  # Target variable

# Scale the features and target variable to integers
scaling_factor = 1000  # Adjust this factor based on your data
X = (X * scaling_factor).astype(int)  # Scale features to integers
y = (y * scaling_factor).astype(int)  # Scale target variable to integers

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=41)

# Use only the first 10 values of X_test
X_test = X_test.head(15)  # Select the first 10 rows for testing
y_test = y_test.head(15)   # Corresponding target values

# Standardize the features (optional, but may not be necessary for integer data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled features back to integers
X_train_scaled = (X_train_scaled * scaling_factor).astype(int)
X_test_scaled = (X_test_scaled * scaling_factor).astype(int)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Get the coefficients (weights) and intercept (bias)
weights = model.coef_
bias = model.intercept_

# Initialize BFV evaluator with parameters from BFV_demo.py
t = 16
n = 1024
q = 132120577
psi = 73993
mu = 0
sigma = 0.5 * 3.2

# Set T to a valid value (make sure it's not zero)
T = 256  # Example value, adjust as necessary

# Generate polynomial arithmetic tables
psiv = modinv(psi, q)
w = pow(psi, 2, q)
wv = modinv(w, q)

# Generate polynomial arithmetic tables
w_table = [1] * n
wv_table = [1] * n
psi_table = [1] * n
psiv_table = [1] * n
for i in range(1, n):
    w_table[i] = (w_table[i - 1] * w) % q
    wv_table[i] = (wv_table[i - 1] * wv) % q
    psi_table[i] = (psi_table[i - 1] * psi) % q
    psiv_table[i] = (psiv_table[i - 1] * psiv) % q

qnp = [w_table, wv_table, psi_table, psiv_table]

# Create the BFV evaluator
Evaluator = BFV(n, q, t, mu, sigma, qnp)

# Generate keys
Evaluator.SecretKeyGen()
Evaluator.PublicKeyGen()

# Generate evaluation keys
Evaluator.EvalKeyGenV1(T)  # Ensure this is called to populate rlk1

# Set the T value in the evaluator
Evaluator.T = T  # Ensure T is set in the evaluator if it's not done in the constructor

print("Encrypting test data...")
# Print the values being encrypted
print("Values to be encrypted (before conversion):", X_test_scaled.flatten())  # Print the flattened values before encryption

# Encrypt the test data
encrypted_X_test = [Evaluator.Encryption(Evaluator.IntEncode(int(value))) for value in X_test_scaled.flatten()]
print("Test data encrypted.")

# Perform predictions on encrypted data without encrypting weights and bias
print("Performing predictions on encrypted data...")

# Initialize a list to hold encrypted predictions for each instance
encrypted_predictions = []

# Define the function for homomorphic multiplication
def perform_homomorphic_multiplication(ct1, ct2):
    # Multiply two messages (with relinearization v1)
    ct = Evaluator.HomomorphicMultiplication(ct1, ct2)
    ct = Evaluator.RelinearizationV1(ct)  # Call to RelinearizationV1
    return ct

# Loop through each instance in the test set
for j in range(len(encrypted_X_test) // X_test.shape[1]):  # Adjusted to loop through instances
    # Start with the encrypted bias
    encrypted_prediction = Evaluator.Encryption(Evaluator.IntEncode(int(bias * scaling_factor)))  # Encrypt the bias only
    
    # Loop through the number of features
    for i in range(X_test.shape[1]):  # X_test.shape[1] gives the number of features
        # Convert weight to an integer or scale it appropriately
        weight_as_int = int(weights[i] * scaling_factor)  # Scale the weight if necessary
        encrypted_weight = Evaluator.Encryption(Evaluator.IntEncode(weight_as_int))  # Encrypt the weight
        
        # Use the correct encrypted feature for the current instance
        encrypted_feature = encrypted_X_test[j * X_test.shape[1] + i]  # Access the correct encrypted feature
        
        # Multiply encrypted feature by the encrypted weight
        encrypted_feature_weight_product = perform_homomorphic_multiplication(encrypted_feature, encrypted_weight)
        
        # Add to the encrypted prediction
        encrypted_prediction = Evaluator.HomomorphicAddition(encrypted_prediction, encrypted_feature_weight_product)
    
    # Store the encrypted prediction for this instance
    encrypted_predictions.append(encrypted_prediction)

print("Predictions completed.")

# Decrypt all predictions
decrypted_predictions = [Evaluator.IntDecode(Evaluator.Decryption(pred)) for pred in encrypted_predictions]

# Scale the decrypted predictions back to the original range
decrypted_predictions_scaled = [pred // scaling_factor for pred in decrypted_predictions]

# Evaluate the model on unencrypted data for comparison
y_pred = model.predict(X_test_scaled)

# Print the results
print("\nDecrypted Predictions for the test instances (Encrypted Data):", decrypted_predictions_scaled)
print("\nPrediction from Unencrypted Model:", y_pred)  # Print predictions for all 10 test instances
print("\nThe actual values: ", y_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print("\nAccuracy Score for Unencrypted Data:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

mae = mean_absolute_error(y_test, decrypted_predictions_scaled)
mse = mean_squared_error(y_test, decrypted_predictions_scaled)
r2 = r2_score(y_test, decrypted_predictions_scaled)

print("\nAccuracy Score for Encrypted Data:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.4f}")