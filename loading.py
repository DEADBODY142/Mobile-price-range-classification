import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the training and testing data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.head())
print(train_data.info())


# print(train_data.head())# Check for missing values
print(train_data.isnull().sum())

# Display basic statistics of the features
print(train_data.describe())

# Correlation matrix
correlation_matrix = train_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Mobile Features')
plt.show()

# Distribution of price ranges
plt.figure(figsize=(8, 6))
sns.countplot(x='price_range', data=train_data)
plt.title('Distribution of Price Ranges')
plt.show()

# Pairplot of some important features
sns.pairplot(train_data[['battery_power', 'ram', 'px_height', 'px_width', 'price_range']], hue='price_range')
plt.show()

print(train_data.info())

# Separate features and target variable
X = train_data.drop('price_range', axis=1)
y = train_data['price_range']

# Create a new feature: total_pixels
X['total_pixels'] = X['px_height'] * X['px_width']

# Create a new feature: pixel_density
X['pixel_density'] = X['total_pixels'] / (X['sc_h'] * X['sc_w'])

from sklearn.impute import SimpleImputer

# Identify numeric columns
numeric_columns = X.select_dtypes(include=[np.number]).columns

# Replace infinite values with NaN
X[numeric_columns] = X[numeric_columns].replace([np.inf, -np.inf], np.nan)

# Impute NaN values with mean
imputer = SimpleImputer(strategy='mean')
X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

# Now scale the numeric columns
X_scaled = X.copy()
scaler=StandardScaler()
X_scaled[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# Now proceed with splitting the data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


print("Shape of training set:", X_train.shape)
print("Shape of validation set:", X_val.shape)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = rf_classifier.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_classifier.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Prepare the test data
X_test = test_data.copy()
X_test['total_pixels'] = X_test['px_height'] * X_test['px_width']
X_test['pixel_density'] = X_test['total_pixels'] / (X_test['sc_h'] * X_test['sc_w'])

# Replace infinite values with NaN in test data
X_test[numeric_columns] = X_test[numeric_columns].replace([np.inf, -np.inf], np.nan)

# Impute NaN values in test data
X_test[numeric_columns] = imputer.transform(X_test[numeric_columns])

# Scale the test data
X_test_scaled = scaler.transform(X_test[numeric_columns])

# Make predictions on the test set
test_predictions = rf_classifier.predict(X_test_scaled)

# Create a DataFrame with the predictions
results = pd.DataFrame({'id': test_data['id'], 'price_range': test_predictions})

# Save the results to a CSV file
results.to_csv('predictions.csv', index=False)

print("Predictions have been saved to 'predictions.csv'")
