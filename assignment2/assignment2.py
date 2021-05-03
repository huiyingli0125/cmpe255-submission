import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def load_data():
    faces = fetch_lfw_people(min_faces_per_person=60)
    n_samples, h, w = faces.images.shape
    X = faces.data
    y = faces.target
    target_names = faces.target_names
    #print (target_names.shape[0])
    return X, y, target_names, h, w

def get_model():
    pca = PCA(n_components=150, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)
    #print(sorted(model.get_params().keys()))
    return model

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model):
    param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005] }
    clf = GridSearchCV(model, param_grid)
    clf = clf.fit(X_train, y_train)
    print (f'Best estimator found by grid search: {clf.best_estimator_}')
    return clf 

def predict(clf, X_test, y_test, target_names):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=target_names))
    return y_pred

def draw_subplots(X_test, y_test, y_pred, h, w, n_row=4, n_col=6):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    #plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(X_test[i].reshape((h, w)), cmap=plt.cm.gray)
        color = "black"
        if y_test[i] != y_pred[i]:
            color = "red"
        plt.xlabel(target_names[y_pred[i]].rsplit(' ', 1)[-1], size=12, color=color)
        plt.xticks(())
        plt.yticks(())
    plt.show()

def draw_confu_matrix(y_test, y_pred):
    ax= plt.subplot()
    cf_matrix = confusion_matrix(y_pred, y_test)
    print (cf_matrix)
    sns.heatmap(cf_matrix, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Confusion Matrix')
    plt.show() 

X, y, target_names, h, w= load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
model = get_model()
clf = train_model(X_train, y_train, model)
y_pred = predict(clf, X_test, y_test, target_names)
draw_subplots(X_test, y_test, y_pred, h, w)
draw_confu_matrix(y_test, y_pred)