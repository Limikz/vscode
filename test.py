from re import T
import numpy as np
import pandas as pd

inputfile = r"C:\Users\Limik\Desktop\数据挖掘\data.csv"
data = pd.read_csv(inputfile)
# description = [data.min(),data.max(),data.mean(),data.std()]
# description = pd.DataFrame(description,index = ['Min','Max','Mean','Std']).T
# print(description)
# corr = data.corr(method='pearson')
# print('相关系数矩阵：\n',np.round(corr,2))

# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.subplots(figsize=(10,10))
# sns.heatmap(corr,annot=True,vmax=1,square=True,cmap="Reds")
# plt.title('相关性热力图')
# plt.show()
# plt.close

from sklearn.linear_model import Lasso

lasso = Lasso(1000)
X,Y = data.drop('y',axis=1),data['y']
lasso.fit(X,Y)
mask = lasso.coef_!=0
# print(lasso.coef_)
# print(mask)

outputfile = r"C:\Users\Limik\Desktop\数据挖掘\news.csv"
new_reg_data = X.iloc[:,mask]
# print(new_reg_data)
# new_reg_data.to_csv(outputfile)
# print(new_reg_data.shape)

from GM11 import GM11

# inputfile2 = r"C:\Users\Limik\Desktop\数据挖掘\news.csv"
# new_data = pd.read_csv(inputfile2)
new_reg_data.index = range(1994,2014)
new_reg_data.loc[2014] = None
new_reg_data.loc[2015] = None
cols = new_reg_data.columns
for i in cols:
    f=GM11(new_reg_data.loc[range(1994,2014),i].values)[0]
    new_reg_data.loc[2014,i] = f(len(new_reg_data)-1)
    new_reg_data.loc[2015,i] = f(len(new_reg_data))
    new_reg_data[i] = new_reg_data[i].round(2)
y = data['y']
y.index = range(1994,2014)
new_data=pd.concat([new_reg_data,y],axis=1)
# print(new_data)

import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR

data_train = new_data.loc[range(1994,2014)].copy()
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train-data_mean)/data_std
x_train,y_train = data_train.drop('y',axis=1),data['y']

linearsvr=LinearSVR()
linearsvr.fit(x_train,y_train)
x = ((new_data[cols]-data_mean[cols])/data_std[cols])
print(x)
y_pred = linearsvr.predict(x,cols)
print(y_pred)

# new_data[u'y_pred'] = linearsvr.predict(x)*data_std['y']+data_mean['y']
# print(new_data[['y','y_pred']])








