import matplotlib.pyplot as plt
from RegLin import *

np.random.seed(0)
x = np.random.rand(100)
y = -2 + 6 * x + np.random.rand(100)

# plot the dataset
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

m = np.size(x)
x_ones = np.ones(len(x))
x_train = np.vstack((x_ones, x)).T

newLin = Linreg()
newLin.fit(x_train, y)
inter = newLin.theta
print(inter)
