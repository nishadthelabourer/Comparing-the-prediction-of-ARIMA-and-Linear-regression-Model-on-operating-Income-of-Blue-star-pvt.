#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[34]:


blue  = pd.read_excel("C:/Users/admin/Videos/cleaned data.xlsx")


# In[39]:


blue.head(2)


# In[36]:


blue.shape


# In[42]:


plt.figure(figsize=(18,5))
plt.plot(blue.Year, blue.operationalIncome, marker='*', color='red', label='Operating Income')
plt.plot(blue.Year, blue.netprofitfortheperiod, marker='*', color='yellow', label='Net Profit of the period')
plt.plot(blue.Year, blue.employeescost, marker='*', color='blue', label='Employee Cost')
plt.plot(blue.Year, blue.totalexpenditure, marker='*', color='black', label='Total Expenditure ')
plt.xlabel('Years')
plt.ylabel('(INR crore)')
plt.legend()
plt.title('Blue Star Operating Income and Profit across the period')
plt.grid()
plt.savefig("C:/Users/admin/Videos/bluestar_graph", dpi=300)


# In[43]:


df= blue[['Year','operationalIncome']]


# In[44]:


df.head(2)


# In[46]:


from statsmodels.tsa.stattools import adfuller


# In[51]:


x= df['operationalIncome'].values
result = adfuller(x)
print('ADF Statistics: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[54]:


## adf statistic value = t-test value of same time series ols model
## becasue my adf statistic value is greater than -3.290 at 5% confidence level i fail to reject my null hypothesis
## that my time series is stationary


# In[57]:


result = adfuller(blue.operationalIncome.diff().dropna())
print('ADF Statistics: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[58]:


## now see my adf statistic value is less than the value at 10% and 5% critical level


# In[59]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[87]:


fig, ax=plt.subplots(1,2, sharex=True, figsize=(10,5))
ax[0].plot(df.operationalIncome); ax[0].set_title('Original Series')
ax[1].set(ylim=(-1,1))
plot_acf(df.operationalIncome, ax=ax[1])
plt.tight_layout()
plt.suptitle('Operating Income')
plt.savefig('C:/Users/admin/Videos/autocorrelation.png', dpi=300)


# In[92]:


fig, ax=plt.subplots(1,2, sharex=True, figsize=(12,5))
ax[0].plot(df.operationalIncome.diff()); ax[0].set_title('Lagged d=1')
ax[1].set(ylim=(-1,1))
plot_acf(df.operationalIncome.diff().dropna(), ax=ax[1])
plt.tight_layout()
plt.suptitle('Operating Income')
plt.savefig('C:/Users/admin/Videos/autocorrelationd=1.png', dpi=300)


# In[93]:


df.shape


# In[106]:


df['Ma4'] = df['operationalIncome'].rolling(window=4).mean()


# In[109]:


df.head(4)


# In[110]:


fig, ax=plt.subplots(1,2, sharex=True, figsize=(12,5))
ax[0].plot(df.Ma4); ax[0].set_title('Ma=4')
ax[1].set(ylim=(-1,1))
plot_pacf(df.Ma4.dropna(), ax=ax[1])
plt.tight_layout()
plt.suptitle('Operating Income')
plt.savefig('C:/Users/admin/Videos/partialautocorrelationd.png', dpi=300)


# In[116]:


df = df.drop([ 'Ma',],1)


# In[182]:


df.tail(5)


# In[118]:


from statsmodels.tsa.arima_model import ARIMA


# In[124]:


x = df.drop(['Year'],1)


# In[125]:


train = x[0:10]


# In[171]:


test2 = x[10:]


# In[172]:


test2.size, train.size


# In[173]:


model_arima = ARIMA(train, order=(0,2,0))
model_arima_fit = model_arima.fit()


# In[174]:


predict = model_arima_fit.forecast(steps=5)[0]


# In[175]:


from sklearn.metrics import mean_absolute_error


# In[178]:


model_error = mean_absolute_error(predict,test2)


# In[179]:


np.sqrt(model_error)


# In[180]:


test2


# In[181]:


predict = pd.DataFrame(data=predict,columns=['ARIMA(0,2,0)'])
forecast = pd.concat([test2,predict],axis=1)
forecast.to_excel("C:/Users/admin/Videos/ARIMA_model.xlsx")


# In[136]:


import itertools 
p=d=q=range(0,5)
pdq=list(itertools.product(p,d,q))


# In[137]:


import warnings 
warnings.filterwarnings('ignore')
for param in pdq:
    try:
        model_arima = ARIMA(train, order=param)
        model_arima_fit = model_arima.fit()
        print(param, model_arima_fit.aic)
    except:
        continue


# In[183]:


print(model_arima_fit.summary())


# In[190]:


data = pd.read_excel("C:/Users/admin/Videos/ARIMA_model.xlsx")


# In[191]:


data.head(2)


# In[195]:


plt.figure(figsize=(18,5))
plt.plot(data.Year, data.operationalIncome, marker='*', color='red', label='Operational Income')
plt.plot(data.Year,data.ARIMA, marker='*', color='green', label='ARIMA(0,2,0)')
plt.xlabel('Years')
plt.ylabel('INR crores')
plt.xticks(data.Year[::1])
plt.title('Predicted Model using Arima model')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("C:/Users/admin/Videos/Graph.png",dpi=300)        


# In[206]:


sns.distplot(blue.operationalIncome, color='red', label='Blue star ltd')
sns.set_style('dark')
plt.gca().set(title=('Distribution plot of Operational Income'))
plt.legend()
plt.savefig("C:/Users/admin/Videos/distribution_plot.png",dpi=300)


# In[208]:


blue.head(2)


# In[214]:


plt.figure(figsize=(10,5))
plt.scatter(blue.Year, blue.rawmaterialconsumption, color='violet', label='Raw Material Consumption')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Raw Material Consumption (INR crores)')
plt.title('Raw Material Consumption by Blue Star')
plt.grid()
plt.savefig("C:/Users/admin/Videos/raw_material_scatter_plot.png",dpi=300)


# In[220]:


datab = blue[['operationalIncome','rawmaterialconsumption','employeescost','netprofitbeforetax','Depreciation','otherexpenses']]


# In[221]:


datab.head(3)


# In[226]:


sns.pairplot(datab)
sns.set_style('dark')
plt.savefig("C:/Users/admin/Videos/pair_plots.png",dpi=300)


# In[234]:


corrmat = datab.corr()


# In[236]:


corrmat.to_excel("C:/Users/admin/Videos/correlation.xlsx")


# In[248]:


fig, ax=plt.subplots(figsize=(10,10))
sns.heatmap(corrmat, linewidth=2, linecolor='red', annot=True)
plt.savefig("C:/Users/admin/Videos/heatmap.png",dpi=300)


# In[249]:


corrmat


# In[250]:


datab.head(2)


# In[252]:


x = np.array(datab.drop(['netprofitbeforetax'],1))


# In[254]:


z = datab[['netprofitbeforetax']]


# In[255]:


y  = np.array(z)


# In[269]:


y


# In[261]:


x.shape,y.size


# In[263]:


from sklearn.model_selection import train_test_split


# In[283]:


x_train,x_test, y_train,y_test = train_test_split(x,y, test_size=0.35,random_state=0)


# In[284]:


from sklearn.linear_model import LinearRegression


# In[2]:


lr = LinearRegression()
lr_fit = lr.fit(x_train, y_train)


# In[1]:


confidence = lr.score(x_test,y_test)
print('Linear regression confidence level:', confidence)


# In[294]:


y_predict = lr.predict(x_test)
y_predict


# In[295]:


x


# In[ ]:




