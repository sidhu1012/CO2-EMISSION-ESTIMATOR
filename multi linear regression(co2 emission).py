import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

df=pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")
df.head()

cdf=df[['ENGINESIZE' , 'CYLINDERS' , 'FUELCONSUMPTION_COMB' , 'CO2EMISSIONS' ]]
cdf.head(9)


cdf.hist()
plt.show()



plt.scatter(cdf.ENGINESIZE ,cdf.CO2EMISSIONS, color='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("EMISSIONS")
plt.show()
plt.scatter(cdf.CYLINDERS , cdf.CO2EMISSIONS,color='red')
plt.xlabel("CYLINDERS")
plt.ylabel("EMISSIONS")
plt.show()
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='green')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("EMISSIONS")
plt.show()



msk=np.random.rand(len(df))<0.8
train=cdf[msk]
test=cdf[~msk]


plt.scatter(train.ENGINESIZE , train.CO2EMISSIONS)
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()



from sklearn import linear_model
regr= linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE','FUELCONSUMPTION_COMB']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x , train_y)
print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_)


y_hat=regr.predict(test[['ENGINESIZE','FUELCONSUMPTION_COMB']])
x=np.asanyarray(test[['ENGINESIZE','FUELCONSUMPTION_COMB']])
y=np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares %.2f"
     % np.mean(y_hat-y)**2)
print("Variance score %.2f"
     % regr.score(x,y))

