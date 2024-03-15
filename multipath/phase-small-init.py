import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng

plt.rc('font', size=20)
from scipy.interpolate import interp1d

rng = default_rng(seed=49827345)


# For small init, we have p ** 2, instead of p ** 2 + 1 (for large init),
# since q ** 2 - p ** 2 = 0 is the constant of motion
def K_full(p, D):
    return p * np.power(p ** 2, (D - 1) / 2)


def get_init_k(n, s=0.1):
    return rng.normal(0, s, size=(n, 2))


def k_trajectory(D, S, p0, def_start, def_end):
    eps = 0.005
    px0, py0 = (eps, eps)
    dt = 0.01

    T = 1000000

    px = np.zeros(T)
    py = np.zeros(T)

    px[0] = px0
    py[0] = py0

    Kx = np.zeros(T)
    Ky = np.zeros(T)

    Kx[0] = K_full(px0, D)
    Ky[0] = K_full(py0, D)

    for t in range(1, T):

        Omega = S - Kx[t - 1] - Ky[t - 1]
        px[t] = px[t - 1] + dt * (np.sqrt(px[t - 1] ** 2) ** (D - 1)) * Omega

        # Added by MK
        if t >= def_start and t < def_end:
            py[t] = py[t - 1]
        else:
            py[t] = py[t - 1] + dt * (np.sqrt(py[t - 1] ** 2) ** (D - 1)) * Omega

        Kx[t] = K_full(px[t], D)
        Ky[t] = K_full(py[t], D)

    return Kx, Ky


kx = np.arange(0.01, 10.01, 0.5)
ky = np.arange(0.01, 10.01, 0.5)

KX, KY = np.meshgrid(kx, ky)
p = np.linspace(0, 10, 10000)


def k_vector_field(D, S):
    K = K_full(p, D)
    pOfK = interp1d(K, p)
    PX = pOfK(KX)
    PY = pOfK(KY)

    Omega = S - KX - KY

    # This comes from K = p * (p^2)^((D-1)/2)
    # and dp = (p^2)^((D-1)/2) * Omega
    dkx = ((KX / PX) ** 2 + (D - 1) * PX * ((KX / PX) ** ((D - 3) / (D - 1))) * KX) * Omega
    dky = ((KY / PY) ** 2 + (D - 1) * PY * ((KY / PY) ** ((D - 3) / (D - 1))) * KY) * Omega

    return dkx, dky


S = 10.
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(6.5 * 3, 6.5 * 2))

std = 0.01
for di, (def_start, def_end) in enumerate([(0, 5), (500000, 500000 + 10000)]):
    D_vals = [2, 3, 4]  # [2,3,8,13]
    p_init_array = get_init_k(1, s=std)
    for i, D in enumerate(D_vals):
        for p_init in p_init_array:
            ax[di, i].set_ylim([0, 10])
            ax[di, i].set_xlim([0, 10])
            k_traj = k_trajectory(D, S, p_init, def_start, def_end)
            ax[di, i].plot(k_traj[0], k_traj[1], 'k', linewidth='5')

        dkx, dky = k_vector_field(D, S)
        ax[di, i].plot(kx, S - kx, 'b-')
        ax[di, i].streamplot(KX, KY, dkx, dky, density=1.0, linewidth=None, color='#A23BEC')
        if di == 0:
            ax[di, i].set_title('$D={}$'.format(D))

ax[0, 0].set_xlabel(r'Path A Singular Value ($K_a$)')
ax[0, 0].set_ylabel(r'Path B Singular Value ($K_b$)')
ax[1, 0].set_xlabel(r'Path A Singular Value ($K_a$)')
ax[1, 0].set_ylabel(r'Path B Singular Value ($K_b$)')

# Create a new text object for the second line of the y-axis label
text_obj = ax[0, 0].annotate('Early Deficit', xy=(0, 0), xytext=(-0.25, 0.5), textcoords='axes fraction',
                             fontsize=30, rotation='vertical', va='center')
text_obj1 = ax[1, 0].annotate('Late Deficit', xy=(0, 0), xytext=(-0.25, 0.5), textcoords='axes fraction',
                              fontsize=30, rotation='vertical', va='center')

fig.tight_layout()
fig.savefig(f"plots/Early-and-Late-Deficit-small-vals.pdf")
