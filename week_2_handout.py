import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

print(np.array([0, 0.5, 1, 1.5, 2, 2.5, 3]))
print(np.arange(0, 3.1, 0.5))
print(np.linspace(0, 3, 7))

x = [1, 2, 3, 4];
y = [5.5, 7.0, 9.5, 9.9];

# for i in range(1, 4):
#     p = np.polyfit(x, y, i)
#
#     x1 = np.linspace(0,5,100)
#     y1 = np.polyval(p,x1)
#
#     plt.plot(x,y,'x')
#     plt.plot(x1,y1,'-')
#
#     plt.show()

tc = np.array([16, 20, 22, 24])
tf = np.array([60, 67, 73, 79])

p = np.polyfit(tc, tf, 2)
x1 = np.linspace(15, 25, 100)
y1 = np.polyval(p, x1)
plt.plot(tc, tf, "x")
plt.plot(x1, y1, "-")
plt.show()
plt.close()

# Tyne bridge modelling
x = [0, 81, 162];
y = [0, 55, 0];

p = np.polyfit(x, y, 2)

x1 = np.linspace(0, 163, 100)
y1 = np.polyval(p, x1)
plt.axes().set_aspect('equal')
plt.plot(x, y, "x")
plt.plot(x1, y1, "-")
plt.show()

# non-polynomial plotting
x = [1, 3, 5, 7, 9]
y = [2.1, 2.9, 3.5, 3.8, 4.2]

plt.loglog(x, y, "x")  # using loglog function
plt.xlabel('x')
plt.ylabel('y')
plt.show()

X = np.log(x)  # not using loglog function (same result)
Y = np.log(y)
plt.plot(X, Y, 'x')
plt.xlabel('X = log(x)')
plt.ylabel('Y = log(y)')

# plt.show()

p = np.polyfit(np.log(x), np.log(y), 1)
x1 = np.linspace(0, 3, 100)
y1 = np.polyval(p, x1)

# Plot
plt.plot(X, Y, 'x')
plt.plot(x1, y1, '-')

plt.xlabel('X = log(x)')
plt.ylabel('Y = log(y)')

plt.show()

x1 = np.linspace(0, 10, 100)  # plotting original data

plt.plot(x, y, 'x')
plt.plot(x1, 2.087 * pow(x1, 0.314), '-')  # y = 2.087x^(0.314)

plt.xlabel('x')
plt.ylabel('y')

plt.show()


# fitting more functions with SciPy
def func(x, a, b, c):
    return a * np.sin(b * x) + c  # returns in the form y = a * sin(b * x) + c


x = np.arange(0, 2, 0.1)
y = np.array([1, 2, 4, 5, 6, 6, 5, 5, 4, 3, 2, 0, -1, -1, -3, -2, -4, -3, -1, 0])

popt, pcov = opt.curve_fit(func, x, y)
print(popt)

x1 = np.linspace(0, 2, 100)
y1 = func(x1, popt[0], popt[1], popt[2])
plt.plot(x, y, 'x')
plt.plot(x1, y1, '-')
plt.show()


# another curve using SciPy to find x(t) = A * e^(k * t) * cos(b * t)
def func2(t, a, b, k):
    return a * np.exp(k * t) * np.cos(b * t)


t = np.arange(0, 2.5, 0.25) # The data
x = np.array([16, 6, -8, -8, 5, 12, 5, -5, -5, 4])

plt.plot(t, x, 'x', markersize=10)  # Plot the data

param, param_cov = opt.curve_fit(func2, t, x)   # Print the best fit parameters
print(param)

t1 = np.linspace(0, 3, 100) # Construct a function to fit
x1 = func2(t1, param[0], param[1], param[2])

plt.plot(t1, x1, '-')

plt.xlabel('t')
plt.ylabel('x')
plt.legend(['Data', 'Best fit x(t)=$Ae^{kt}\cos(bt)$'], loc=1)
plt.show()
