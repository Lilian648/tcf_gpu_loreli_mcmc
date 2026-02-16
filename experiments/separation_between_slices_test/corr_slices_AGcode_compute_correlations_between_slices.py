import matplotlib.pyplot as plt
import numpy as np

N=256
# make a random cube for the example
simcube = np.random.normal(0, 1, size=(N, N, N))


L=160.  # side length of the simulation
deltax = L/N
x = np.arange(N//2)*deltax  # physical distance between two slices


# cross correlations
corr = np.zeros(N//2)
# loop over distance between slices (n=0 = auto corr)
for n in range(N//2):
    # multiply each slice with the following ones, wrapping around cube edges as it is periodic
    diag = np.array([simcube[j]*np.take(simcube, j+n, mode='wrap') for j in range(N)])
    # diag has shape (N, N, N)
    # average over all the pairs of sims separated by distance n
    corr[n] = np.mean(diag) - np.mean(simcube)**2

plt.figure(figsize=(9,8))
plt.axhline(0,color='k',ls=':')    
plt.plot(x, corr, lw=2.)
plt.ylabel(r'$V(s)$', fontsize=18)
plt.xlabel(r'Slice separation $s$ [Mpc]', fontsize=18)
plt.xlim(left=0)
plt.tight_layout()
