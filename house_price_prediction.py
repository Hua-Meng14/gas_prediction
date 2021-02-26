import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("gas_data .csv")
model = LinearRegression()
x = df["Distant"]
y = df["Gas"]
x=np.array(x).reshape(-1,1)
y=np.array(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
model.fit(x_train,y_train)
print(model.predict([[124],[345]]))
print(model.score(x_test,y_test))
