import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === Step 1: Load features and labels ===
features_df = pd.read_csv('finger_features.csv')  # Input features per frame
labels_df = pd.read_csv('labels.csv')             # Manual labels: 1 = press, 0 = no press

# === Step 2: Merge data based on frame number ===
data = pd.merge(features_df, labels_df, on='frame')

# === Step 3: Prepare training inputs and targets ===
X = data.drop(columns=['frame', 'label'])  # Input features
y = data['label']                          # Target label

# === Step 4: Split into train and test sets ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 5: Train classifier (Random Forest) ===
model = RandomForestClassifier(
    n_estimators=100,                         # Number of trees
    class_weight='balanced',                  # Handle class imbalance
    random_state=42,                          # For reproducibility
    n_jobs=-1                                 # Use all CPU cores
)
model.fit(X_train, y_train)

# === Step 6: Evaluate performance ===
y_pred = model.predict(X_test)

print("=== Classification Report (Random Forest) ===")
print(classification_report(y_test, y_pred, digits=4))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# === Step 7: Save trained model to file ===
joblib.dump(model, 'press_classifier.pkl')
print(" Model saved as press_classifier.pkl")
