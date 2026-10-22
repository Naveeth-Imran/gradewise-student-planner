import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Load the dataset
data = pd.read_csv("students.csv")

print("Student Dataset:")
print(data.head())

# 2. Separate input and output
X = data[["Hours"]]
y = data["Marks"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict marks for test data
predictions = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print("\nMean Absolute Error:", round(mae, 2))

# 7. Save the trained model
with open("student_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")

# 8. Plot the dataset and regression line
plt.scatter(X["Hours"], y, label="Actual marks")

plt.plot(
    X["Hours"],
    model.predict(X),
    color="red",
    label="Regression line"
)

plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Student Marks Prediction")
plt.legend()
plt.grid(True)
plt.savefig("prediction_graph.png")
plt.show()
