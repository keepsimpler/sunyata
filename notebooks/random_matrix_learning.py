# %%
import numpy as np
import matplotlib.pyplot as plt
# %%
n = 5000
A = np.random.randn(n, n)
# Gaussian Orthogonal Ensemble
GOE = (A + A.T) / np.sqrt(2*n)

eig = np.linalg.eigvals(GOE)
plt.hist(eig, bins=100, density=True)
# %%
n = 5000
r = 1/2
X = np.random.randn(int(n * r), n)
W = np.dot(X, X.T) / n

eig = np.linalg.eigvals(W)
plt.hist(eig, bins=100, density=True)
# %%
n = 1000
A = np.random.randn(n, n) / np.sqrt(n)
eig = np.linalg.eigvals(A)
plt.scatter(np.real(eig), np.imag(eig))
plt.axes().set_aspect(1)
plt.show()
# %%
n = 1000
M = np.random.random((n,n)) - 0.5
M *= (12/n) ** 0.5
w = np.linalg.eigvals(M)
plt.scatter(np.real(w), np.imag(w))
plt.axes().set_aspect(1)
plt.show()
# %%
from scipy.stats import ortho_group 

n = 1000
X = ortho_group.rvs(n)
eig = np.linalg.eigvals(X)
plt.scatter(np.real(eig), np.imag(eig))
plt.axes().set_aspect(1)
plt.show()

# %%
