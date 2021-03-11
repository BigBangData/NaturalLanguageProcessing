from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

# LR 
@timer
def train_LR(X_train, y_train):

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("svm_clf", LogisticRegression(solver="saga", random_state=42, max_iter=1000, n_jobs=-1))
    ])
    clf.fit(X_train, y_train)
    return clf

# SVM 
@timer
def train_SVC(X_train, y_train):

    poly_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler(with_mean=False)),
        ("svm_clf", LinearSVC(loss="hinge", random_state=42, tol=1e-5, max_iter=1000))
    ])
    poly_svm_clf.fit(X_train, y_train)
    return poly_svm_clf

poly_svm_clf = train_SVC(X_train, y_train)

# Predict the response for test dataset
y_pred = poly_svm_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# persisting model
from pathlib import Path

# make dir if not exists, including parent dirs
dirpath = os.path.join("..","data","4_models","sentiment140")
Path(dirpath).mkdir(parents=True, exist_ok=True)

# save model 
now = str(int(time.time()))
filename = ''.join([now, "_poly_svm_clf_lemm_bow_1pctsample.joblib"])
filepath = os.path.join(dirpath, filename)

from joblib import dump, load
dump(poly_svm_clf, filepath)

# load pre-trained model 
dirpath = os.path.join("..","data","4_models","sentiment140")
os.listdir(dirpath)

filename = os.listdir(dirpath)[0]
filepath = os.path.join(dirpath, filename)
poly_svm_clf = load(filepath)