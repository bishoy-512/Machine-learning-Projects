import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv(r"D:\Machine Learning Projects\Titanic\train.csv")

# Drop unused columns
df.drop(columns=['Age', 'Name', 'Parch', 'PassengerId', 'SibSp', 'Cabin' , 'Ticket'], inplace=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

num_col = X.select_dtypes(exclude='object').columns
cat_col = X.select_dtypes(include='object').columns

# Numeric preprocessing pipeline
num_pip = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing pipeline
cat_pip = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocess = ColumnTransformer([
    ('num', num_pip, num_col),
    ('cat', cat_pip, cat_col)
])

# Full pipeline with model
model = Pipeline([
    ('preprocess', preprocess),
    ('model', RandomForestClassifier(random_state=42))
])

# Hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 300],
    'model__max_depth': [5, 7],
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Grid search
grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Accuracy
print("Accuracy:", accuracy_score(y_test, best_model.predict(X_test)) * 100)

# Save model
import os
model_path = os.path.join(os.getcwd(), "model.pkl")

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"Model saved successfully at: {model_path}")

