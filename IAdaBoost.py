import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.metrics import geometric_mean_score
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

class IAdaBoostClassifier:
    def __init__(self, n_estimators=50, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []
        self.alphas = []
        self.classes_ = None

    def fit(self, X, y):
        # Get the class labels
        self.classes_ = np.unique(y)
        N = len(y)
        class_counts = Counter(y)
        class_weights = {k: N / v for k, v in class_counts.items()}

        # Initialize the sample weights
        w = np.array([class_weights[yi] for yi in y])
        w /= w.sum()

        for t in range(self.n_estimators):
            # Train the weak classifier
            model = DecisionTreeClassifier(max_depth=self.max_depth)
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            # Calculate the error rate
            incorrect = (y_pred != y).astype(int)
            error = np.dot(w, incorrect) / w.sum()

            # Prevent division by zero errors
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Calculate the model weights
            alpha = 0.5 * np.log((1 - error) / error)
            self.models.append(model)
            self.alphas.append(alpha)

            # Update the sample weights and incorporate the class weights
            w = w * np.exp(alpha * incorrect) * np.array([class_weights[yi] for yi in y])
            w /= w.sum()

    def predict(self, X):
        # Initialize the score matrix
        scores = np.zeros((X.shape[0], len(self.classes_)))

        for alpha, model in zip(self.alphas, self.models):
            y_pred = model.predict(X)
            for idx, cls in enumerate(self.classes_):
                scores[:, idx] += alpha * (y_pred == cls)

        # Select the class with the highest score
        y_pred = self.classes_[np.argmax(scores, axis=1)]
        return y_pred

# Data processing and model training
if __name__ == "__main__":
    # Data reading
    data = np.genfromtxt("")
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]

    # Data normalization
    mm = MinMaxScaler()
    X = mm.fit_transform(X)

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)

    # Encode labels as integers
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    classes = le.classes_

    # Initialize and train a custom IAdaBoost model
    model = IAdaBoostClassifier(n_estimators="optimal_parameter_tuning_for_n_estimators", max_depth="optimal_parameter_tuning_for_max_depth")
    model.fit(X_train, y_train_encoded)
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Calculate the G-Mean
    gms = geometric_mean_score(y_test_encoded, y_pred, average='macro')
    print("G-Mean of the test set:", gms)

