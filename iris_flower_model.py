## Load libraries
from pandas import read_csv # to real csv file to be loaded in an object
from pandas.plotting import scatter_matrix # need this to plot an object in scatter matrix
from matplotlib import pyplot # need this draw plot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

## Load dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'];
# Attributes information can be read here https://archive.ics.uci.edu/ml/datasets/Iris
# This will be treated as header of each column in .csv file
dataset = read_csv('iris.csv', names = names)

## Summarize the dataset
# Print dim of dataset via shape // 'shape' is a field of object 'dataset'
print("Number of rows, number of columns: ")
print(dataset.shape) # (150, 5) Each flower in 150 flowers (not species) has 5 attributes

# Print first 5 lines of 'dataset' via head() // 'head()' is a method of object 'dataset'
print("Fist 5 lines: ")
print(dataset.head(5))

# Print data distribution of 'dataset' via describe() // 'describe()' is a method of object 'dataset'
print("Data distribution: ")
print(dataset.describe()) # count = number of values for each attributes, mean = average, std = standart deviation

# Print class distribution
print("Class distribution: ")
print(dataset.groupby('class').size()) # Group all flowers in a class then print each class's total number

## Visualize data
# Histogram via hist() // 'hist()' is a method of object 'dataset'
#dataset.hist()
#pyplot.show()

# Scatter matrix via pandas.plotting.scatter_matrix()
scatter_matrix(dataset)
pyplot.show()