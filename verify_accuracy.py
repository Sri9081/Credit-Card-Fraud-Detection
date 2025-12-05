import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import time

def verify():
    print("Loading data...")
    try:
        df = pd.read_csv("creditcard.csv")
    except FileNotFoundError:
        print("Error: creditcard.csv not found.")
        return

    print(f"Data loaded: {df.shape}")

    # Preprocessing
    if 'Time' in df.columns:
        df = df.drop('Time', axis=1)
    
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"Resampled shape: {X_train_res.shape}")
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy', n_jobs=-1, random_state=42)
    start = time.time()
    rf.fit(X_train_res, y_train_res)
    print(f"Training time: {time.time() - start:.2f}s")
    
    print("Evaluating...")
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    if acc > 0.95:
        print("SUCCESS: Accuracy > 95%")
    else:
        print("FAILURE: Accuracy < 95%")

if __name__ == "__main__":
    verify()
