"""
This script shows an example of how to use the tcf code
with batch computation.

Authors: Nicolas Monnier & Adélie Gorce
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import cupy as cp
from cupyx.scipy import special


def window_function(x, ndim=2):
    if ndim == 2:
        return special.j0(x)
    elif ndim == 3:
        return cp.sinc(x / cp.pi) * cp.pi  # sinc(x) = sin(pi x) / (pi x)
    else:
        raise ValueError('Only supports ndim=2 or 3.')


def tcf_partial_vectorized_gpu(rlin, Bk_values, k_norms, q_norms, p_norms, L, ndim):
    r = cp.array(rlin[:, cp.newaxis] , dtype=cp.float32)         # shape = (len(rlin), 1)
    p = p_norms[cp.newaxis, :]       # shape = (1, len(Bk_values))

    masks = (k_norms[cp.newaxis, :] <= cp.pi/r) & (q_norms[cp.newaxis, :] <= cp.pi/r)

    weights_gpu = (r / L)**(3.*ndim/2.) * window_function(p*r, ndim)
    sums = cp.sum(weights_gpu * masks * Bk_values[cp.newaxis, :], axis=1)

    return sums


def compute_gpu_norms(samples, ndim, kcoord_gpu):
    """Computes |k|, |q|, |k+q| norms for each bispectrum triplet"""
    kvecs = samples[:, 0:ndim]
    qvecs = samples[:, ndim:2*ndim]

    kvecs_gpu = cp.array(kvecs, dtype=cp.int16)
    qvecs_gpu = cp.array(qvecs, dtype=cp.int16)

    k_selected = cp.take(kcoord_gpu, kvecs_gpu, axis=0)
    q_selected = cp.take(kcoord_gpu, qvecs_gpu, axis=0)

    k_norm = cp.linalg.norm(k_selected, axis=1)
    q_norm = cp.linalg.norm(q_selected, axis=1)

    px = k_selected[:,0] + 0.5*q_selected[:,0] + cp.sqrt(3)/2 * q_selected[:,1]
    py = k_selected[:,1] - cp.sqrt(3)/2 * q_selected[:,0] + 0.5*q_selected[:,1]
    if ndim==2:
        pz = np.zeros_like(px)
    elif ndim == 3:
        pz = k_selected[:,2] + q_selected[:,2]
    p_norm = cp.linalg.norm(cp.array([px, py, pz], dtype=cp.float32), axis=0)

    return k_norm, q_norm, p_norm


def estimate_bispectrum_gpu(epsilon_k, n, samples_gpu, ndim):
    """
    Monte Carlo estimator for the bispectrum.
    Returns an array of bispectrum values corresponding to each (k, q, p=k+q).
    """

    kvecs = samples_gpu[:, 0:ndim].get()
    qvecs = samples_gpu[:, ndim:2*ndim].get()
    pvecs = (kvecs + qvecs) % n

    if ndim == 2:
        ek = epsilon_k[kvecs[:,0], kvecs[:,1]]
        eq = epsilon_k[qvecs[:,0], qvecs[:,1]]
        ep = epsilon_k[pvecs[:,0], pvecs[:,1]]    
    else:
        ek = epsilon_k[kvecs[:,0], kvecs[:,1], kvecs[:,2]]
        eq = epsilon_k[qvecs[:,0], qvecs[:,1], qvecs[:,2]]
        ep = epsilon_k[pvecs[:,0], pvecs[:,1], pvecs[:,2]]    

    return ek * eq * cp.conj(ep)


def generate_kq_pairs_batches_gpu(n, ndim, batch_size=1000000):
    """
    Générateur qui produit des blocs de paires (kx, ky, qx, qy)
    de taille contrôlée par `batch_size`.

    Cela parcourt linéairement l’espace n^4 sans jamais créer le tableau complet.
    """
    total = n**(ndim*2)
    print(f"Nombre total de paires : {total:,}")
    print(f"Nombre de batches : {total//batch_size:,}")

    total_pairs = n**(ndim*2)
    count = 0

    for start in range(0, total_pairs, batch_size):
        end = min(start + batch_size, total_pairs)
        size = end - start

        # Indices linéaires dans l’espace (kx, ky, qx, qy)
        flat_idx = cp.arange(start, end, dtype=np.int64)

        # Décodage des indices 2*ndim D à partir d’un index linéaire

        if ndim == 3:
            qz = flat_idx % n
            flat_idx //= n
        qy = flat_idx % n
        flat_idx //= n
        qx = flat_idx % n
        flat_idx //= n
        if ndim == 3:
            kz = flat_idx % n
            flat_idx //= n
        ky = flat_idx % n
        kx = flat_idx // n
        if ndim == 2:
            pairs = cp.stack([kx, ky, qx, qy], axis=1).astype(np.int16)
            yield pairs
        else:
            triplet = cp.stack([kx, ky, kz, qx, qy, qz], axis=1).astype(cp.int16)
            yield triplets

        count += size

    print(f"Nombre total de paires après batching : {count:,}")
