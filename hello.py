import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data (replace this with your actual servo data)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Input feature (e.g., voltage)
y = 2 * X + 1 + np.random.randn(100, 1)  # Output (e.g., servo position)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('Input (e.g., Voltage)')
plt.ylabel('Output (e.g., Servo Position)')
plt.title('Servo Prediction using Linear Regression')
plt.legend()
plt.show()

# Function to predict servo position for a given input
def predict_servo_position(input_value):
    return model.predict([[input_value]])[0][0]

# Example usage
input_value = 5.0
predicted_position = predict_servo_position(input_value)
print(f"Predicted servo position for input {input_value}: {predicted_position:.2f}")