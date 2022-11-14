import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt 

inputfile = r"C:\Users\Limik\Desktop\数据挖掘\air_data.csv"
data = pd.read_csv(inputfile)

# explore = data.describe(percentiles=[],include='all').T
# explore['null'] = len(data)-explore['count']
# explore = explore[['null','max','min',]]
# print(explore)

ffp = data['FFP_DATE'].apply(lambda x:datetime.strptime(x,'%Y/%m/%d'))
ffp_year = ffp.map(lambda x:x.year)
# print(ffp_year)
# fig = plt.figure(figsize=(8,5))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# plt.hist(ffp_year,bins='auto',color='#0504aa')
# plt.xlabel('年份')
# plt.title('各年份会员入会人数')
# plt.ylabel('入会人数')
# plt.show()
# plt.close

# male = pd.value_counts(data['GENDER'])['男']
# female = pd.value_counts(data['GENDER'])['女']
# fig = plt.figure(figsize=(7,4))
# plt.pie([male,female],explode=[0.01,0.01],labels=['男','女'],colors=['lightskyblue','lightcoral'],autopct='%1.1f%%')
# plt.title('会员性别比例')
# plt.show()
# plt.close

# lv_four = pd.value_counts(data['FFP_TIER'])[4]
# lv_five = pd.value_counts(data['FFP_TIER'])[5]
# lv_six = pd.value_counts(data['FFP_TIER'])[6]

# fig = plt.figure(figsize=(8,5))
# x=[4,5,6]
# x_label=['lv_four','lv_five','lv_six']
# plt.bar(x,height=[lv_four,lv_five,lv_six],width=0.4,color = 'skyblue')
# plt.xticks(x,x_label)
# plt.xlabel('会员数量')
# plt.ylabel('会员人数')
# plt.title('会员各级别人数')
# plt.show()
# plt.close()

# age = data['AGE'].dropna()
# age = age.astype(int)
# fig = plt.figure(figsize=(5,10))
# plt.boxplot(age,patch_artist=True,labels=['会员年龄'],boxprops={'facecolor':'lightblue'})
# plt.title('会员年龄分布箱线图')
# plt.grid(axis='both')
# plt.show()
# plt.close()

# data_corr = data[['FFP_TIER','FLIGHT_COUNT','LAST_TO_END','SEG_KM_SUM','EXCHANGE_COUNT','Points_Sum']]
# age1 = data['AGE'].fillna(0)
# data_corr['AGE'] = age1.astype(int)
# data_corr['ffp_year'] = ffp_year

# dt_corr = data_corr.corr(method='pearson')
# print('相关系数矩阵：\n',dt_corr)

# import seaborn as sns
# plt.subplots(figsize=(8,8))
# sns.heatmap(dt_corr,annot=True,square=True,cmap='Blues')
# plt.show()
# plt.close()

# print(data.shape)
# data_notnull = data.loc[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull()]
# print(data_notnull.shape)

a = ['1','2','3']
b = ['4','5','6']
a={'a':a}
b={'b':b}
data_a = pd.DataFrame(a)
data_b = pd.DataFrame(b)
data_a=pd.concat([data_a,data_b],axis=1)
data_a.iloc[1,0]=np.NAN
data_a.iloc[2,1]=np.NaN
# print(data_a.loc[1,'a'])
print(data_a.loc[1].isnull()['a'])
#     print('ok')
# data_a=data_a.loc[data_a['a'].notnull() & data_a['b'].notnull()]
print(data_a)






