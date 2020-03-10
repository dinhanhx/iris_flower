## Load libraries
from pandas import read_csv # to real csv file to be loaded in an object
from pandas.plotting import scatter_matrix # need this to plot an object in scatter matrix
from matplotlib import pyplot # need this draw plot
from sklearn.model_selection import train_test_split # to create train and validation dataset
# Accuracy estimations
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
# Modles
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Save models
import pickle as pk

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
#scatter_matrix(dataset)
#pyplot.show()

## Create validation dataset
arr = dataset.values
X = arr[:,0:4] # Get all values from col 0 to 3
Y = arr[:,4] # Get all values at col 4
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = 0.20, random_state = 1) # 20% to test, 80% to train, split randomly with random_state = 1

## Input models
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

## Evaluate model in turn
print('Model, mean score, standard deviation: ')
results = []
name_of_models = []
for name, model in models:
	k_fold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
	cross_val_results = cross_val_score(model, X_train, Y_train, cv = k_fold, scoring = 'accuracy')
	results.append(cross_val_results)
	name_of_models.append(model)
	print('%s: %f (%f)' % (name, cross_val_results.mean(), cross_val_results.std()))

print('Chose Support Vector Machines model')

## Make predictions
# Create chose model
chose_model = SVC(gamma = 'auto')
name_of_model = 'Supprot Vector Machine'

# Give it foods and request from it
chose_model.fit(X_train, Y_train)
Y_predictions = chose_model.predict(X_validation)

# Print out the accuracy score
print('Accuracy score: ')
print(accuracy_score(Y_validation, Y_predictions))

## Save the chose model
pk.dump(chose_model, open('iris_flower_prophet.pkl', 'wb'))
