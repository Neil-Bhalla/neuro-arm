import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

actions = {"blink": 0, "bite": 1, "double_blink": 2, "double_bite": 3}
data, labels = [], []

for action, label in actions.items():
    action_features = np.load(f'{action}_features.npy')
    data.append(action_features)
    labels.append(np.full(action_features.shape[0], label))

X = np.concatenate(data, axis=0)
y = np.concatenate(labels, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(kernel='linear')
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=actions.keys()))

dump(clf, 'svm_classifier.joblib')
dump(scaler, 'scaler.joblib')
