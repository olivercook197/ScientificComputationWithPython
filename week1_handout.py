import numpy as np
import matplotlib.pyplot as plt

"""this week we created graphs, learnt about polyfit which automatically gives a value of m and c in y = mx + c"""


# create a graph, -2 <= x < 2, against x, x^2 and x^3
x = np.linspace(-2, 2, 100)
g = x ** 2
h = x ** 3

plt.plot(x, x)
plt.plot(x, g)
plt.plot(x, h)

plt.show()

plt.close()

# create a graph, x, against sin(x)
x = np.linspace(0,10,100)
y = np.sin(x)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('sin(x)')

plt.show()

plt.close()

# curve fitting by trial and error
x = np.arange(1, 5)
y = [5.5, 7.0, 9.5, 9.9]
plt.plot(x, y, 'x')

x1 = np.linspace(0, 5, 100)
f = 3 * x / 2 + 4
plt.plot(x, f, "-")

plt.xlabel('x')
plt.ylabel('y')
plt.show()
r = f - y
print(r)
S = sum(pow(r, 2))
print(S)
plt.close()

# curve fitting using polyfit
x = np.arange(1, 5)
y = [5.5, 7.0, 9.5, 9.9]
plt.plot(x, y, 'x')

x = np.arange(1, 5)
y = [5.5,7.0,9.5,9.9]
p = np.polyfit(x, y, 1)
f = p[0] * x + p[1]
plt.plot(x, f)

plt.xlabel('x')
plt.ylabel('y')
plt.show()

r = f - y
print(r)
S = sum(pow(r, 2))
print(S)
plt.close()
print(p)

plt.close()

x = np.arange(1, 10)
y = [13, 11, 10, 9, 7, 6, 4, 3, 2]
plt.plot(x, y, 'x')

p = np.polyfit(x, y, 1)
f = p[0] * x + p[1]
plt.plot(x, f)

plt.xlabel('x')
plt.ylabel('y')
plt.show()

r = f - y
print(r)
S = sum(pow(r, 2))
print(S)
plt.close()
print(p)

