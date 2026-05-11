import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load your dataset
# df = pd.read_csv('loan_data.csv')

# For demonstration, let's assume columns: Gender, Married, Education, Income, LoanAmount, Credit_History
# Dummy data creation for structure
data = {
    'Gender': [1, 0, 1, 1], # 1: Male, 0: Female
    'Married': [1, 1, 0, 1],
    'Education': [1, 1, 0, 1], # 1: Graduate, 0: Not
    'Income': [5000, 2000, 6000, 3000],
    'Credit_History': [1, 0, 1, 1],
    'Loan_Status': [1, 0, 1, 1]
}
df = pd.DataFrame(data)

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")