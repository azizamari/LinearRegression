import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data=pandas.read_csv(r'salary_data.csv',delimiter=',')
scaler = StandardScaler()
data=scaler.fit_transform(data)
print(data)

def cost_function(points,m,b):
    n=len(points)
    total_error=0
    for i in range(n):
        x=points[i,0]
        y=points[i,1]
        total_error+=(m*x+b-y)**2
    total_error/=2*n
    return total_error
def gradient_descent(m_now,b_now,points,l):
    m_gradient=0
    b_gradient=0
    n=len(points)
    for i in range(n):
        x=points[i,0]
        y=points[i,1]
        m_gradient+=(m*x+b-y)*x
        b_gradient+=(m*x+b-y)
    m_now=m_now-(1/n*m_gradient)*l
    b_now=b_now-(1/n*b_gradient)*l
    return m_now,b_now

b=0
m=0
epochs=400
for i in range(epochs):
    m,b=gradient_descent(m,b,data,0.01)
    if i%50==0:
        print('epochs: ',i)
        print('cost: ', cost_function(data,m,b))
plt.scatter(data[:,0],data[:,1])
plt.plot(list(range(1,10)),[m*x+b for x in range(1,10)])
plt.show()