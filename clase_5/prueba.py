import numpy as np
import pandas as pd
import gradient_descent
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from models import LinearRegressionNumpy

x_data = np.linspace(0,1,40)
noise = 1*np.random.uniform(  size = 40)
y_data = np.sin(x_data * 1.5* np.pi )
y_data = (y_data + noise-1).reshape(-1,1)


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(15, include_bias=False)
poly_data = poly.fit_transform(x_data.reshape(-1, 1))

colname = ['x']
for i in range(2, 16):
    colname.append('x_%d'%i)

colname.append('y')
data = pd.DataFrame(np.column_stack([poly_data,y_data]),columns=colname)

pipe = Pipeline(steps=[
           ('select', StandardScaler()),
           ('clf', LinearRegressionNumpy())
        ])


x = data.iloc[:,0:1]

y = np.reshape(data.y.values, (-1,1))
pipe.fit(x,y)

X_expanded = np.hstack((x, np.ones((len(x),1))))


lr_1 = 0.0001
# Set up the n° of epochs
amt_epochs_1 = 100


W_manual = gradient_descent.gradient_descent(X_expanded, y, lr=lr_1, amt_epochs=amt_epochs_1)
print (W_manual)



# modelo = LinearRegressionB()
#
#
# modelo.fit(np.array(X_train), np.array(Y_train))
#
# y_predicted = modelo.predict(np.array(X_test).reshape(-1, 1))
#
# print("Error cuadrático medio mi modelo: ", metrics.mean_squared_error(Y_test, y_predicted))



# modelo2 = linear_model.LinearRegression()
# modelo2.fit(np.array(X_train).reshape(-1, 1), Y_train)
# y_predicted2 = modelo2.predict(np.array(X_test).reshape(-1, 1))
#
# print("Error cuadrático medio scikit: ", metrics.mean_squared_error(Y_test, y_predicted2))
#

import os
os.getcwd()




