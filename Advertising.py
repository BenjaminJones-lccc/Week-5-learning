import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("Advertising.csv")

# Independent and dependent variables
X = df[['TV']]
y = df['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Intercept (β₀):", model.intercept_)
print("Slope (β₁):", model.coef_[0])
print("R² Score:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(df['TV'], df['Sales'])
plt.plot(df['TV'], model.predict(df[['TV']]), linewidth=2)
plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")
plt.show()