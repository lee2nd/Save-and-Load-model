# https://scikit-learn.org/stable/model_persistence.html

# pickle (and joblib by extension), has some issues regarding maintainability and security. Because of this,
# 1. Never unpickle untrusted data as it could lead to malicious code being executed upon loading.
# 2. While models saved using one version of scikit-learn might load in other versions, this is entirely unsupported and inadvisable. It should also be kept in mind that operations performed on such data could give different and unexpected results.

from sklearn import datasets
import skops.io as sio
import joblib

X, y = datasets.load_iris(return_X_y=True)
    
# joblib load
with open("svc_model.joblib", 'rb') as f:
    clf = joblib.load(f)    

# check what the model is    
print(clf)

obj = sio.dumps(clf)
clf_sio = sio.loads(obj, trusted=True)

# check result
print('classifier pred:', clf_sio.predict(X))
print('classifier load pred:', clf_sio.predict(X))    
