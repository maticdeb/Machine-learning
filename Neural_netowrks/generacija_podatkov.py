import numpy as np
from scipy.interpolate import interp1d
from numba import njit
import time
import os
from joblib import Parallel, delayed
from typing import Union
import matplotlib.pyplot as plt
import matplotlib
import random


def generate_function(n_points, f0, f1, max_offsets=(1.3, 0.5, 0.3, 0.05)):
    """
    Generates values of a random function f on the interval [0, 1] with boundary conditions f(0)=f0 and f(1)=f1
    by dividing the interval on smaller subsections and iteratively applying random perturbations to function values
    on these subsections, starting from the linear function f. Smoothness of f is achieved with quadratic interpolation.

    :param n_points: desired length of the output (discretization number)
    :param f0: desired value of f(0)
    :param f1: desired value of f(1)
    :param max_offsets: maximal perturbation offsets for each iteration.
        The length of this touple also determines the number of iterations.
    :return: values of the random function in n_points discrete points on the interval [0, 1]
    """

    n_op = n_points // 2  # we operate with half the resolution, which helps the smoothness
    x = np.linspace(0, 1, n_op)
    A = f0 + (f1 - f0) * x

    iterations = len(max_offsets)
    max_num_of_intervals = n_op // (2 ** np.arange(iterations)[::-1])

    for i in range(iterations):
        interval_boundaries = np.sort(np.random.randint(1, n_op - 1, np.random.randint(2, max_num_of_intervals[i])))
        interval_lenghts = interval_boundaries[1:] - interval_boundaries[:-1]
        interval_offsets = np.random.uniform(-max_offsets[i], max_offsets[i], size=(len(interval_boundaries)-1,))
        offsets = np.repeat(interval_offsets, interval_lenghts)
        A[interval_boundaries[0]:interval_boundaries[-1]] += offsets

    indeksi0 = np.sort(np.random.permutation(np.arange(1, n_op - 1))[:np.random.randint(1, n_op // 8)])
    indeksi = np.concatenate(([0], indeksi0, [n_op - 1]))

    f = interp1d(x[indeksi], A[indeksi], kind='quadratic')

    return f(np.linspace(0, 1, n_points))


def intensity(theta_evolution, lbd=1000 * 1e-9, D=15 * 1e-6, n_o0=1.53, n_e0=1.71):
    """
    Calculates time dependent intensity of transmitted linearly polarized light.

    :param theta_evolution: Time evolution of theta profile (array of dim (n_timesteps, n_zsteps))
    :param lbd: wavelength in meters
    :param D: thickness of the layer in meters
    :param n_o0: ordinary refractive index
    :param n_e0: extraordinary refractive index
    :return: intensity time dependence (array of length n_timesteps)
    """

    n_zsteps = theta_evolution.shape[1]
    h = D / (n_zsteps - 1)

    theta_evolution = (theta_evolution[:, :-1] + theta_evolution[:, 1:]) / 2

    n_e = (1 / ((np.cos(theta_evolution)) ** 2 / n_e0 ** 2 + (np.sin(theta_evolution)) ** 2 / n_o0 ** 2)) ** 0.5
    dPHI0 = np.sum((2 * np.pi / lbd) * h * (n_e - n_o0), axis=1)

    return (np.sin(dPHI0 / 2)) **  2

def theta_time_evolution(theta0, C, D=15 * 1e-6, dt=2 * 1e-8, num_timesteps=200000, nth_step_save=500):
    """
    Calculates time evolution (relaxation) of director profile (given by angle theta).

    :param theta0: Starting profile of theta (array of length N)
    :param C: Relaxation constant
    :param D: Thickness of the layer in meters
    :param dt: Timestep in seconds
    :param num_timesteps: Number of timesteps in the simulation
    :param nth_step_save: Save theta profile at every nth step (number of saves: M = num_timesteps // nth_step_save)
    :return: Time evolution of theta (array of dimensions (M, N))
    """

    N = len(theta0)
    dz = D / (N - 1)
    cnst = C * dt / (dz ** 2)

    if cnst > 0.5:
        raise ValueError("Iteration step too large, try smaller timestep or change other parameters.")

    thetas_out = np.zeros((num_timesteps // nth_step_save, N))
    theta1 = np.copy(theta0)
    for t in range(num_timesteps):
        theta1[1:N-1] += cnst * (theta1[2:N] - 2 * theta1[1:N-1] + theta1[:N-2])
        if t % nth_step_save == 0:
            thetas_out[t // nth_step_save, :] = theta1

    return thetas_out      

z = np.linspace(0,15,400)
t = np.linspace(0,2*10**(-8)*200000,400)

for k in range(200):
    konst = np.random.uniform(0, 1)*3*10**(-8)
    a = generate_function(400,0,0)
    b = theta_time_evolution(theta0=a,C=konst)
    c= intensity(b)
    cmap = plt.get_cmap('gnuplot')
    number=len(b)
    colors = [cmap(i) for i in np.linspace(0, 1, number)]
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(7,4))
    for i, color in enumerate(colors, start=1):
        ax1.plot(z,b[i-1], color=color)
        ax1.set(xlabel="z [\u03BCm]",ylabel="\u03B8") 
    ax2.plot(t,c)
    ax2.text(0.0025, 0.9, f'C = {round(konst,11)}', fontsize=10)

    ax2.set(xlabel='t[s]',ylabel='I')
    plt.show()

