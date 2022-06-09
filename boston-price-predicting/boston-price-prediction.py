# -*- coding: utf-8 -*-

# importing dependencies 
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston = load_boston()

df_x = pd.DataFrame(data = boston.data, columns = boston.feature_names)

df_y = pd.DataFrame(boston.target)

regression = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(df_x , df_y,test_size = 0.33 , random_state = 42)

regression.fit(x_train,y_train)

y_prediction = regression.predict(x_test)


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test,y_prediction)











