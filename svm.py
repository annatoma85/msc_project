import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
clf = svm.SVC(gamma = 0.001, C=100)

print(len(digits.data))

print(digits.data)

print(digits.target)

print(digits.image[0])

