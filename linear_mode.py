import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import datasets , linear_model

x = datasets.load_diabetes()
# print(data.keys())
data_x = x.data[:,np.newaxis,2]
# data_x = x.data  # all data mentioned here we predict value
# print(data_x)
data_x_trained = data_x[:-50]
data_x_test = data_x[-50:]
# print(data_x_trained,data_x_test)

data_y_trained = x.target[:-50]
data_y_test = x.target[-50:]

model = linear_model.LinearRegression()
model.fit(data_x_trained,data_y_trained)

data_predict = model.predict(data_x_test)
print("mean squred error : ",mean_squared_error(data_y_test,data_predict))

print("Weight is : ",model.coef_)
print("Interception : ",model.intercept_)

plt.scatter(data_x_test,data_y_test)
plt.plot(data_x_test,data_predict)
plt.show()