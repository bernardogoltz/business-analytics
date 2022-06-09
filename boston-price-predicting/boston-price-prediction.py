# -*- coding: utf-8 -*-

# importing dependencies 
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

boston = load_boston()

df_x = pd.DataFrame(data = boston.data, columns = boston.feature_names)

df_y = pd.DataFrame(boston.target)

regression = linear_model.LinearRegression()