import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)

data = {
    "links": [0, 5, 7, 9],
    "length": [20, 30, 40, 50],
    "spam": [0, 0, 1, 1]
}

df = pd.DataFrame(data)
print(df)

X = df[["links", "length"]]
Y = df["spam"]

model = LogisticRegression()
model.fit(X, Y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Spam email check
spam_email = pd.DataFrame([[11, 60]], columns=["links", "length"])
print("\nSpam email prediction:")
print(model.predict(spam_email))
print("Spam email probabilities:")
print(model.predict_proba(spam_email))

# Normal email check
normal_email = pd.DataFrame([[1, 15]], columns=["links", "length"])
print("\nNormal email prediction:")
print(model.predict(normal_email))
print("Normal email probabilities:")
print(model.predict_proba(normal_email))

# Middle email check
middle_email = pd.DataFrame([[6, 35]], columns=["links", "length"])
print("\nMiddle email prediction:")
print(model.predict(middle_email))
print("Middle email probabilities:")
print(model.predict_proba(middle_email))

# Evaluation on the training data
Y_pred = model.predict(X)

print("\nConfusion Matrix:")
print(confusion_matrix(Y, Y_pred))

print("\nPrecision:", precision_score(Y, Y_pred, zero_division=0))
print("Recall:", recall_score(Y, Y_pred, zero_division=0))

print("\nClassification Report:")
