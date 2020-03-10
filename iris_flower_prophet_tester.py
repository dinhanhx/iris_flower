import pickle as pk
from pandas import read_csv

model = pk.load(open('iris_flower_prophet.pkl', 'rb'))

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'];
dataset = read_csv('iris.csv', names = names)

arr = dataset.values
X = arr[0:1, 0:4]
Y = arr[0:1, 4]

print('iris_flower_prophet said: ')
Y_predicted = model.predict(X)
print(Y_predicted)

print('human said: ')
print(Y)

## if the answers are different
# Give it new food and save it
#model.fit(X, Y)
#pk.dump(model, open('iris_flower_prophet.pkl', 'wb'))
