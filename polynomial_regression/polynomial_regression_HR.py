import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

f,n= map(int, input().split())
x=[]
y=[]
for i in range(n):
    val=list(map(float,input().split()))
    x.append(val[:-1])
    y.append(val[-1])
n_t=int(input())
x_test=[]
for i in range(n_t):
    x_test.append(list(map(float,input().split())))


poly=PolynomialFeatures(degree=3) #degree is less than 3
X=poly.fit_transform(x)
X_test=poly.fit_transform(x_test)

model=LinearRegression()
model.fit(X,y)

p=model.predict(X_test)
for i in p:
    print(i)