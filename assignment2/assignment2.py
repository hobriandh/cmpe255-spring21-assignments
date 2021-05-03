from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import random

def load_data():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print('data loaded')
    print(faces.target_names)
    print(faces.images.shape)
    return faces


def grid_search(X_train, y_train, X_test, y_test):
    pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)

    param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__gamma':[1, 0.1, 0.01, 0.001, 0.0001], 'svc__gamma': ['scale', 'auto']}

    grid = GridSearchCV(model, param_grid)
    grid.fit(X_train, y_train) 

    print(grid.best_params_) 
    grid_predictions = grid.predict(X_test) 

    print(classification_report(y_test, grid_predictions)) 
    return grid_predictions

def plot(X_test, y_test, predictions, target_names):
    fig, axes = plt.subplots(4, 6, figsize = (12, 12))
    for i in range(4):
      for j in range(6):
          random_face = random.randint(0, 100)
          pred = predictions[random_face]
          img = X_test[random_face]
          true = y_test[random_face]
          axes[i, j].imshow(img.reshape(62, 47))
          if pred == true:
            axes[i, j].set_title(target_names[true], color = 'black')
          if pred != true:
            axes[i, j].set_title(target_names[true], color = 'red')
    plt.savefig('plot.png')
    plt.show()
    pass

faces = load_data()

# split data
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size = 0.2)
print(X_train.shape)
print(X_test.shape)

# use grid search
predictions = grid_search(X_train, y_train, X_test, y_test)

# plot the faces
plot(X_test, y_test, predictions, faces.target_names)
print('savings faces to: ', 'plots.png')

# get heatmap
'''
cm = metrics.confusion_matrix(predictions, y_test, labels = faces.target_names)
sns.heatmap(cm, annot=True)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show(block = True)
'''