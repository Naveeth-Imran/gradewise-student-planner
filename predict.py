import pickle

# Load the trained model
with open("student_model.pkl", "rb") as file:
    model = pickle.load(file)

print("=== Student Marks Prediction System ===")

hours = float(input("Enter the number of hours studied: "))

if hours < 0:
    print("Study hours cannot be negative.")
else:
    predicted_marks = model.predict([[hours]])[0]

    # Keep the result within the 0-100 marks range
    predicted_marks = max(0, min(100, predicted_marks))

    print("Hours Studied:", hours)
    print("Predicted Marks:", round(predicted_marks, 2), "/ 100")
