import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Create a synthetic dataset
data = {'Feature1': [1.2, 2.4, 3.1, 4.2, 5.5],
        'Feature2': [0.9, 1.8, 2.7, 3.6, 4.5],
        'Real_Effort_Person_Hours': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)

# Split the dataset into X (features) and y (target)
X = df.drop(columns=["Real_Effort_Person_Hours"])
y = df["Real_Effort_Person_Hours"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
clf = SVC(kernel='linear')

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
