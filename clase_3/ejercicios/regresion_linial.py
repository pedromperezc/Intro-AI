from clase_3.ejercicios.models import LinearRegressionB
from clase_3.ejercicios.set_datos import SetDatos
import numpy as np
from sklearn import metrics, linear_model

data = SetDatos("data/income.csv")

train, validation, test = data.split()

X_train, Y_train = zip(*train)
X_test, Y_test = zip(*test)
modelo = LinearRegressionB()


modelo.fit(np.array(X_train), np.array(Y_train))

y_predicted = modelo.predict(np.array(X_test))

print("Error cuadrático medio mi modelo: ", metrics.mean_squared_error(Y_test, y_predicted))
print (modelo.coef_)
print (modelo.intercept_)


modelo2 = linear_model.LinearRegression()
modelo2.fit(np.array(X_train).reshape(-1, 1), Y_train)
y_predicted2 = modelo2.predict(np.array(X_test).reshape(-1, 1))

print("Error cuadrático medio scikit: ", metrics.mean_squared_error(Y_test, y_predicted2))

print (modelo2.coef_)
print (modelo2.intercept_)



