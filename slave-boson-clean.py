
import numpy as np
import matplotlib.pyplot as plt

def fock_operators(n_max):
    # Local operators in Fock space
    D = n_max + 1
    b = np.zeros((D, D), dtype=float)
    for n in range(1, D):
        b[n-1, n] = np.sqrt(n)
    n_op = np.diag(np.arange(D, dtype=float))
    I = np.eye(D, dtype=float)
    return {"b": b, "n": n_op, "I": I}

def mf_hamiltonian(phi, J, U, mu, z, ops):
    # the mean-field local Hamiltonian
    b, n_op = ops["b"], ops["n"]
    nnm1 = n_op @ (n_op - np.eye(n_op.shape[0]))
    h = 0.5*U*nnm1 - mu*n_op - z*J*(phi*b.T + phi*b)
    return h

def solve_mf(J, U, mu, z=4, n_max=5, phi0=0.3, tol=1e-12, max_iter=4000, mix=0.5):
    # self-consistent mean-field solution
    ops = fock_operators(n_max)
    b = ops["b"]

    phi = float(phi0)
    last = None
    for _ in range(max_iter):
        h = mf_hamiltonian(phi, J, U, mu, z, ops)
        E, V = np.linalg.eigh(h)
        v0 = V[:, 0]
        phi_raw = float(np.real(v0.T @ b @ v0))
        if phi_raw < 0:
            V[:, 0] *= -1.0
            v0 = V[:, 0]
            phi_raw = float(np.real(v0.T @ b @ v0))
        phi_new = (1 - mix)*phi + mix*phi_raw
        if last is not None and abs(phi_new - phi) < tol:
            phi = phi_new
            break
        last = phi
        phi = phi_new

    # when phi converges
    h = mf_hamiltonian(phi, J, U, mu, z, ops)
    E, V = np.linalg.eigh(h)
    v0 = V[:, 0]
    phi = float(np.real(v0.T @ b @ v0))
    if phi < 0:
        V[:, 0] *= -1.0
        v0 = V[:, 0]
        phi = float(np.real(v0.T @ b @ v0))

    F = V.T @ b @ V  # b operator rotated in mean-field solution basis  
    n_rot = V.T @ ops["n"] @ V
    return {"phi": phi, "E": E, "V": V, "F": F, "n_rot": n_rot,
            "ops": ops, "J": J, "U": U, "mu": mu, "z": z, "n_max": n_max}

def sb_blocks_k(mf, kx, ky):
    # Build A_k and B_k for a given k vector. Neccesarhy to construcrt the quadratic Hamiltonian
    J, z, E, F = mf["J"], mf["z"], mf["E"], mf["F"]
    n_max = mf["n_max"]
    N = n_max

    Delta = E[1:] - E[0]
    F_a0 = F[1:, 0].astype(float)
    F_0a = F[0, 1:].astype(float)

    A0 = np.diag(Delta)
    A1 = - J * (np.outer(F_a0, F_a0) + np.outer(F_0a, F_0a))
    B  = - J * (np.outer(F_0a, F_a0) + np.outer(F_0a, F_a0).T)

    eta_k = 0.5 * (np.cos(kx) + np.cos(ky))
    A_k = A0 + z * eta_k * A1
    B_k = z * eta_k * B
    return A_k, B_k

def bdg_diagonalize(A_k, B_k):
    # Bogoliubov diagonalization of quadratic Hamiltonian to get physical spectrum and eigenmodes U V
    N = A_k.shape[0]

    H = np.block([[A_k, B_k],
                  [B_k, A_k]])
    Jb = np.block([[np.eye(N), np.zeros((N, N))],
                   [np.zeros((N, N)), -np.eye(N)]])
    M = Jb @ H
    w, X = np.linalg.eig(M)
    w = w.real
    idx = np.argsort(w)
    w, X = w[idx], X[:, idx]

    pos = w > 1e-10
    w_pos, X_pos = w[pos], X[:, pos]

    # split into U, V
    U = np.zeros((N, w_pos.size), dtype=complex)
    V = np.zeros((N, w_pos.size), dtype=complex)
    for i in range(w_pos.size):
        x = X_pos[:, i]
        norm = float((x.conj().T @ Jb @ x).real)
        if norm < 0:
            x = -x
            norm = -norm
        x = x / np.sqrt(norm)
        U[:, i] = x[:N]
        V[:, i] = x[N:]
    return w_pos, U, V

def sweep_kx0_to_plot_PS(mf, J, num_points=251, save_path="physical_spectrum.png"):
    ky = np.linspace(-np.pi, np.pi, int(num_points))
    e_low = np.zeros((2, ky.size), dtype=float)

    for j, kyj in enumerate(ky):
        A_k, B_k = sb_blocks_k(mf, 0.0, kyj)
        w_pos, _, _ = bdg_diagonalize(A_k, B_k)
        e = np.sort(w_pos)
        e_low[0, j] = e[0] if e.size > 0 else np.nan
        e_low[1, j] = e[1] if e.size > 1 else np.nan

    plt.figure(figsize=(4, 4))
    plt.plot(ky/np.pi, e_low[0], label=r"$\alpha=1$")
    plt.plot(ky/np.pi, e_low[1], label=r"$\alpha=2$")
    plt.xlabel(r"$k_y$")
    plt.ylabel(r"$\omega_{k,\alpha}$")
    plt.title("Physical Spectrum")
    plt.xticks([-1,-0.5,0,0.5,1], [r'-$\pi$',r'-$\pi$/2','0',r'$\pi/2$' ,r'$\pi$'])
    plt.legend(loc="best")
    plt.ylim( bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    return ky, e_low


def _bdg_u_v(Ak, Bk, tol=1e-10):
    # Bogoliubov diagonalization of quadratic Hamiltonian to get physical spectrum and eigenmodes U V
    N = Ak.shape[0]
    H = np.block([[Ak, Bk],
                  [Bk, Ak]])
    Sigma = np.block([[np.eye(N), np.zeros((N, N))],
                      [np.zeros((N, N)), -np.eye(N)]])

    evals, evecs = np.linalg.eig(Sigma @ H)

    # Keep only positive frequencies
    pos = evals.real > tol
    omegas = evals.real[pos]
    vecs = evecs[:, pos]

    # Sort
    idx = np.argsort(omegas)
    omegas = omegas[idx]
    vecs = vecs[:, idx]

    # Normalization
    for j in range(vecs.shape[1]):
        v = vecs[:, j]
        nrm = v.conj().T @ Sigma @ v
        vecs[:, j] = v / np.sqrt(nrm)

    U = vecs[:N, :]
    V = vecs[N:, :]
    return omegas, U, V

def _sb_correlators_k(mf, kx, ky):
    # Constructs the correlators C11, C12, C22 for specific k vector
    Ak, Bk = sb_blocks_k(mf, kx, ky)
    w, U, V = _bdg_u_v(Ak, Bk)
    C11 = U @ U.conj().T
    C12 = U @ V.T 
    C22 = V @ V.conj().T
    return C11, C12, C22


# build C_A at fixed ky for the half-torus cut and extract ES
def entanglement_spectrum_half_torus(mf, ky, L, kx_points=None, return_n=False):
    #calculates EE for half a torus
    # return_n if i want the occupations

    n_flav = mf["n_max"]  # number of SB flavors
    Lx = 2 * L
    Nkx = int(kx_points) if kx_points is not None else Lx

    # kx grid for the torus of length 2L
    kxs = 2.0 * np.pi * (np.arange(Nkx) - Nkx // 2) / Lx

    # Fourier sums for x
    # We need C11(delta), C12(delta), C22(delta) for delta in [-(L-1), ..., (L-1)]
    deltas = np.arange(-(L-1), L, dtype=int)
    C11_of_delta = {d: np.zeros((n_flav, n_flav), dtype=complex) for d in deltas}
    C12_of_delta = {d: np.zeros((n_flav, n_flav), dtype=complex) for d in deltas}
    C22_of_delta = {d: np.zeros((n_flav, n_flav), dtype=complex) for d in deltas}

    for kx in kxs:
        C11_k, C12_k, C22_k = _sb_correlators_k(mf, float(kx), float(ky))
        for d in deltas:
            phase = np.exp(1j * kx * d)
            C11_of_delta[d] += phase * C11_k
            C12_of_delta[d] += phase * C12_k
            C22_of_delta[d] += phase * C22_k

    # Normalize Fourier transform along x
    for d in deltas:
        C11_of_delta[d] /= Nkx
        C12_of_delta[d] /= Nkx
        C22_of_delta[d] /= Nkx

    # Assemble C_A
    M = L * n_flav
    C11_A = np.zeros((M, M), dtype=complex)
    C12_A = np.zeros((M, M), dtype=complex)
    C22_A = np.zeros((M, M), dtype=complex)

    def idx(x, alpha):  # x in [0..L-1], alpha in [0..n_flav-1]
        return x * n_flav + alpha

    for x in range(L):
        for xp in range(L):
            d = x - xp
            G11 = C11_of_delta[d]
            G12 = C12_of_delta[d]
            G22 = C22_of_delta[d]
            for a in range(n_flav):
                ia = idx(x, a)
                for b in range(n_flav):
                    jb = idx(xp, b)
                    C11_A[ia, jb] = G11[a, b]
                    C12_A[ia, jb] = G12[a, b]
                    C22_A[ia, jb] = G22[a, b]

    # Build the full correlation matrix on A in the gamma slave boson operators basis:
    C_A = np.block([[C11_A,          C12_A],
                    [C12_A.conj(),   C22_A]])

    #Diagonalization of C_A
    # eigenvalues are {1+n_j} and { -n_j }.
    m = M
    Upsilon = np.block([[np.eye(m), np.zeros((m, m))],
                        [np.zeros((m, m)), -np.eye(m)]])
    evals, _ = np.linalg.eig(Upsilon @ C_A)
    # Separate positive and negative sets
    lam_pos = np.sort(evals.real[evals.real > 1e-9])          #  1+n_j
    lam_neg = np.sort(evals.real[evals.real < -1e-9])         # -n_j

    # if numerical imag parts or missing count
    if lam_pos.size == 0 or lam_neg.size == 0:
        raise RuntimeError("Failed to get entanglement occupations: check k-mesh/L.")

    n_from_pos = lam_pos - 1.0
    n_from_neg = -lam_neg[::-1]                               # reverse because easier for me to manage
    n_occ = 0.5 * (n_from_pos[:min(len(n_from_pos), len(n_from_neg))] +
                   n_from_neg[:min(len(n_from_pos), len(n_from_neg))])
    # entanglement spectrum
    omega_A = np.log(1.0 + 1.0 / np.maximum(n_occ, 1e-14))
    omega_A_sorted = np.sort(omega_A.real)
    return (omega_A_sorted, n_occ) if return_n else omega_A_sorted


def plot_entanglement_spectrum_ky(mf, L=50, ky_points=100, branches=6, kx_points=None, save_path=None, show=True):
    kys = np.linspace(-np.pi, np.pi, ky_points)
    curves = [[] for _ in range(branches)]
    for ky in kys:
        wA = entanglement_spectrum_half_torus(mf, float(ky), L, kx_points=kx_points)
        take = min(branches, len(wA))
        for i in range(take):
            curves[i].append(wA[i])
        for i in range(take, branches):
            curves[i].append(np.nan)

    plt.figure(figsize=(4, 4), dpi=130)
    for i in range(branches):
        plt.plot(kys/np.pi, curves[i], lw=1.5)
    #plt.xlim(-np.pi, np.pi)
    plt.xlabel(r"$k_y$")
    plt.ylabel(r"$\omega^{(A)}_{k_y,p}$")
    plt.title("Entanglement spectrum")
    plt.ylim(top=20, bottom=0)
    plt.xticks([-1,-0.5,0,0.5,1], [r'-$\pi$',r'-$\pi$/2','0',r'$\pi/2$' ,r'$\pi$'])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()



def entropy_from_occupations(n_occ):
    n = np.asarray(n_occ, dtype=float)
    n = np.clip(n, 0.0, None)
    # log1p for numerical
    term1 = (1.0 + n) * np.log1p(n)
    # if log(0)
    term2 = np.where(n > 0, n * np.log(n), 0.0)
    return float(np.sum(term1 - term2))


def entanglement_entropy_half_torus_total(mf, L, ky_points=None, kx_points=None, return_per_length=True):

    Ly = L
    Nk = int(ky_points) if ky_points is not None else Ly
    kys = np.linspace(-np.pi, np.pi, Nk)

    S_total = 0.0
    for ky in kys:
        # get entanglement occupations at fixed ky, integrating over kx
        _, n_occ = entanglement_spectrum_half_torus(mf, ky=float(ky), L=L,
                                                    kx_points=kx_points, return_n=True)
        S_total += entropy_from_occupations(n_occ)

    if return_per_length:
        return S_total, S_total / float(Ly)
    else:
        return (S_total,)


def sweep_J_entanglement_entropy(mu, J_min=0.0375, J_max=0.05, num_J=27, U=1.0, z=4, n_max=5, L=48, ky_points=None, kx_points=None, phi0=0.3, tol=1e-7, max_iter=2000, mix=0.5, plot=True, save_path=None):

    Js = np.linspace(J_min, J_max, int(num_J))
    S_total = np.zeros_like(Js, dtype=float)
    S_per_len = np.zeros_like(Js, dtype=float)

    for i, J in enumerate(Js):
        print(i)
        # MF at this J
        mf = solve_mf(J=J, U=U, mu=mu, z=z, n_max=n_max, phi0=phi0, tol=tol, max_iter=max_iter, mix=mix)
        # Entanglement entropy for the half-torus at this J
        S_tot, S_per = entanglement_entropy_half_torus_total(mf, L=L, ky_points=ky_points, kx_points=kx_points, return_per_length=True)
        S_total[i] = S_tot
        S_per_len[i] = S_per
        print(S_tot)

    if plot:
        plt.figure(figsize=(6.2, 3.6), dpi=130)
        plt.plot(Js*z, S_per_len, marker='o', lw=1.6)
        plt.xlabel(r"$Jz$")
        plt.ylabel(r"$S_A/L_A$")
        plt.title("entanglement entropy")
        plt.ylim(top=0.35)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()

    return Js, S_total, S_per_len






def main():
    U = 0.0
    z = 4
    J = (.2)/z
    mu = np.sqrt(2)-1
    n_max =3

    mf = solve_mf(J, U, mu, z, n_max, phi0=0.3, tol=1e-6, max_iter=500, mix=0.5)
    Ak0, Bk0 = sb_blocks_k(mf, 0.0, 0.0)
    wGamma, _, _ = bdg_diagonalize(Ak0, Bk0)

    sweep_kx0_to_plot_PS(mf, J, num_points=101, save_path='spectrum-deep-MI.png')
    print("Saved PS plot")

    #plot_entanglement_spectrum_ky(mf, L=50, ky_points=100, branches=4, kx_points=None, save_path="entanglement_spectrum.png", show=True)
    print("Saved entanglement_spectrum.png")

    L = 50
    ky_points = 2*L
    kx_points = 2*L

    #Js, S_total, S_over_Ly = sweep_J_entanglement_entropy(mu=mu, J_min=0.05, J_max=0.06, num_J=2, U=U, z=z, n_max=n_max, L=L, ky_points=ky_points, kx_points=kx_points, plot=True, save_path="EE2.png")

if __name__ == "__main__":
    main()
