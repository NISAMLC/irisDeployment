import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_iris()

X = pd.DataFrame(data.data, columns=(data.feature_names))
y = pd.DataFrame(data.target, columns=['Target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=1)

def training_model():
    model = DecisionTreeClassifier()
    trained_model = model.fit(X_train, y_train)
    return trained_model