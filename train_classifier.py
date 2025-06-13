"""Train a simple SVM classifier using Apprentissage.mat."""
import pickle
import os
import scipy.io
import numpy as np
from sklearn.svm import SVC
import cv2
from utils.classification import color_histogram


def main(mat_path='Apprentissage.mat', output='metro_classifier.pkl'):
    if not os.path.exists(mat_path):
        print('Training data not found')
        return
    data = scipy.io.loadmat(mat_path)
    images = data['images']  # placeholder names
    labels = data['labels'].ravel()

    X = []
    for img in images:
        patch = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        X.append(color_histogram(patch))
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, labels)
    with open(output, 'wb') as f:
        pickle.dump(clf, f)
    print('Model saved to', output)


if __name__ == '__main__':
    main()
