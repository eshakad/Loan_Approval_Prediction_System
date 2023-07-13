import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the dataset
data = pd.read_csv('loan_data.csv')

# Preprocess the data
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status'].replace({'Y': 1, 'N': 0})
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict loan approval for the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

import pickle
with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)

