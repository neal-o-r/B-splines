import numpy as np
import matplotlib.pyplot as plt

def B_spline(x, n, k):
	'''
	x = Array of x points, 
	n = number of (equally spaced) knots
	k = spline degree
	'''
	if x.ndim == 1:
		x = x.reshape(1,-1)

	xl = np.min(x)
	xr = np.max(x)
	dx = (xr - xl) / n # knot seperation

	t = (xl + dx * np.arange(-k, n+1)).reshape(1,-1)
	T = np.ones_like(x).T * t # knot matrix
	X = x.T * np.ones_like(t) # x matrix
	P = (X - T) / dx # seperation in natural units
	B = ((T <= X) & (X < T+dx)).astype(int) # knot adjacency matrix
	r = np.roll(np.arange(0, t.shape[1]), -1) # knot adjacency mask

	# compute recurrence relation k times
	for ki in range(1, k+1):
		B = (P * B + (ki + 1 - P) * B[:, r]) / ki

	return B


x = np.linspace(0, 10, 101)
n = 10
k = 4
np.random.seed(123)

Bs = B_spline(x, n, k)
A  = np.arange(Bs.shape[1]) + np.sin(np.arange(Bs.shape[1]))
B = Bs.dot(A)

plt.plot(A * Bs, lw=1, alpha=0.5)
plt.plot(B, c='k', lw=1.5)
plt.show()
