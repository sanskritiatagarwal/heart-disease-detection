import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cardio_train dataset with the correct delimiter
data = pd.read_csv('/Users/gauravsingh/Documents/hospital/assign-2/cardio_train.csv', delimiter=';')

# Print column names to debug
print("Columns in the dataset:", data.columns)

# Convert age from days to years
if 'age' in data.columns:
    data['age'] = data['age'] / 365
else:
    raise KeyError("The 'age' column is missing from the dataset.")

# Select relevant columns for cardiovascular health metrics
features = data[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke']]
target = data['cardio']  # Assuming 'cardio' is the target column indicating heart issues

# Handle missing values if any
features.fillna(features.mean(), inplace=True)

# Encode categorical variables if necessary
features = pd.get_dummies(features, columns=['gender', 'cholesterol', 'gluc', 'smoke'], drop_first=True)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()

# Define a more comprehensive parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'max_iter': [100, 200, 300]
}

# Initialize GridSearchCV with both accuracy and precision as scoring metrics
grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring=['accuracy', 'precision'],
    refit='accuracy',  # Choose which metric to optimize
    return_train_score=True
)

# Fit the model
grid.fit(X_train, y_train)

# Best model based on accuracy
best_model = grid.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"Best Parameters: {grid.best_params_}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(classification_report(y_test, y_pred))

# Print detailed results
results = pd.DataFrame(grid.cv_results_)

# Pivot the results DataFrame for visualization
accuracy_pivot = results.pivot_table(
    index='param_C', 
    columns='param_solver', 
    values='mean_test_accuracy'
)

precision_pivot = results.pivot_table(
    index='param_C', 
    columns='param_solver', 
    values='mean_test_precision'
)

# Plot heatmaps for accuracy and precision
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.heatmap(accuracy_pivot, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title('Mean Test Accuracy')
plt.xlabel('Solver')
plt.ylabel('C')

plt.subplot(1, 2, 2)
sns.heatmap(precision_pivot, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title('Mean Test Precision')
plt.xlabel('Solver')
plt.ylabel('C')

plt.tight_layout()
plt.show()