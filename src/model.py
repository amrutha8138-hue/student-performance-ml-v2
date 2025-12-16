import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("../data/student_data.csv")

X = data[['study_hours', 'attendance', 'previous_marks']]
y_marks = data['final_marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_marks, test_size=0.2, random_state=42
)

linear = LinearRegression()
linear.fit(X_train, y_train)
print("Predicted Marks:", linear.predict(X_test))

y_pass = data['pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_pass, test_size=0.2, random_state=42
)

log = LogisticRegression()
log.fit(X_train, y_train)

pred = log.predict(X_test)
print("Pass/Fail Accuracy:", accuracy_score(y_test, pred))
