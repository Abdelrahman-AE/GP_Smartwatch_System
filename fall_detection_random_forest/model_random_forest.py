import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Load the dataset
file_path = 'acc_gyr.csv'
data = pd.read_csv(file_path)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values:\n", missing_values)

# Encode the labels (multi-class)
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Separate features and labels
X = data.drop(columns=['label'])
y = data['label']

# Address class imbalance by oversampling minority classes
df = pd.concat([X, y], axis=1)

# Find the maximum number of samples in any class
max_size = df['label'].value_counts().max()

# Resample each class to the maximum size
lst = [df]
for class_index, group in df.groupby('label'):
    lst.append(group.sample(max_size-len(group), replace=True))
df_balanced = pd.concat(lst)

# Separate features and labels again
X_balanced = df_balanced.drop(columns=['label'])
y_balanced = df_balanced['label']

# Split the balanced data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = rf_clf.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Test with a new value
new_value = [[4.79, 2.47, -2.9, 17.58, -3.91, -16.3]]  # Replace this with your actual new data point
new_value_scaled = scaler.transform(new_value)

# Predict the label for the new data point
new_prediction = rf_clf.predict(new_value_scaled)
predicted_label = label_encoder.inverse_transform(new_prediction)
print("Predicted label for the new value:", predicted_label[0])


import joblib

# Save the Random Forest model
joblib.dump(rf_clf, 'model_random_forest.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')
