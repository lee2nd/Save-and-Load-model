# ref : https://practicaldatascience.co.uk/machine-learning/how-to-save-and-load-machine-learning-models-using-pickle

# Load packages
import pandas as pd
import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Load the data
X, y = load_diabetes(return_X_y=True, as_frame=True)
X.head()

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = XGBClassifier(random_state=0)
model.fit(X_train, y_train)

# Save the model with Pickle
pickle.dump(model, open('model.pkl', 'wb'))

# Load the model from Pickle
pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(X_test)
