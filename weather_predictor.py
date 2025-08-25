import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load and prepare data (same as main.py)
df = pd.read_csv("seattle-weather.csv")
df['date'] = pd.to_datetime(df['date'])

# Create date features
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['season'] = df['month'].map({12:4, 1:4, 2:4, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3})
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Train the best model with proper train/test split
best_features = ['precipitation', 'temp_max', 'temp_min', 'wind', 'month', 'day_of_year', 'season', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
X = df[best_features]
Y = df['weather']

# Split data for training and validation
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123, stratify=Y)

# Train model on training data - Optimized to reduce overfitting
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    max_depth=7,           
    min_samples_split=5,   
    min_samples_leaf=3,    
    max_features=5
)
model.fit(X_train, y_train)

# Evaluate model performance on test set
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Model Training Complete!")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Check training accuracy vs test accuracy
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)

print(f"\n=== TRAIN VS TEST ACCURACY ===")
print(f"Training accuracy:    {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test accuracy:        {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Save the trained model
with open('weather_model.pkl', 'wb') as f:
    pickle.dump(model, f)
