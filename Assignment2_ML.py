import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits() #loading the dataset
clf = svm.SVC(gamma =0.001, C=100) #initializing the classifier
print("dataset size:")
print(len(digits.data))
a,b = digits.data[:-10], digits.target[:-10] #this is the training data and target data
clf.fit(a,b) #Using an in-built svm training function from sklearn

print('Prediction:', clf.predict(digits.data[[-2]]))
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r) #converts array to image.
plt.show() # shows the target image to compare vs the prediction.
