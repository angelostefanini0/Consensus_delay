import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles

# Configurazione sistema
N = 10
delta = 30
T = 1500
alpha = 0.4
k_p = 0
k_v = 2.99
dT = 0.01

# Modello dinamico (double integrator)
I3 = np.eye(3)
Z3 = np.zeros((3, 3))
A = np.block([[I3, dT * I3], [Z3, I3]])
B = np.vstack([0.5 * dT**2 * I3, dT * I3])
K = np.hstack([k_p * I3, k_v * I3])
C = np.hstack([I3, Z3])

# Osservatore
obs_poles = [0.8] * 3 + [0.9] * 3
L_single = place_poles(A.T, C.T, obs_poles).gain_matrix.T
L = np.tile(L_single[None, :, :], (N, 1, 1))

# Topologia (ring)
A_conn = np.zeros((N, N))
for i in range(N):
    A_conn[i, (i - 1) % N] = 1
    A_conn[i, (i + 1) % N] = 1
D_inv = np.diag(1 / np.sum(A_conn, axis=1))
W = D_inv @ A_conn

# Inizializzazione stati
np.random.seed(0)
p0 = np.random.rand(N, 3) * 5
v0 = np.random.rand(N, 3)
x0 = np.hstack([p0, v0])
x_hat0 = x0 + 0.01 * np.random.randn(N, 6)

p_history = [x0]
x_hat_history = [x_hat0]
z_history = [x0.copy() for _ in range(delta + 1)]
u_history = []

# Simulazione
for t in range(T):
    x = p_history[-1]
    x_hat = x_hat_history[-1]
    z = z_history[-1]
    z_d = z_history[-(delta + 1)]

    new_z = (1 - alpha) * z + alpha * (W @ z_d)
    u = - (K @ x_hat.T).T + (K @ new_z.T).T

    new_x = (A @ x.T).T + (B @ u.T).T
    y = (C @ x.T).T
    y_hat = (C @ x_hat.T).T
    new_x_hat = (A @ x_hat.T).T + (B @ u.T).T + np.einsum('nij,nj->ni', L, y - y_hat)

    p_history.append(new_x)
    x_hat_history.append(new_x_hat)
    z_history.append(new_z)
    u_history.append(u)

# Conversione array
p_array = np.array(p_history)
x_hat_array = np.array(x_hat_history)
u_array = np.array(u_history)
positions = p_array[:, :, :3]
velocities = p_array[:, :, 3:]
time = np.arange(T + 1) * dT
u_time = np.arange(T) * dT
error = np.linalg.norm(p_array - x_hat_array, axis=2)

# Componenti x
px = positions[:, :, 0]
vx = velocities[:, :, 0]
ux = u_array[:, :, 0]

# Plot posizione e velocità
fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
for i in range(N):
    axs[0].plot(time, px[:, i])
    axs[1].plot(time, vx[:, i])
axs[0].set_title('Componente posizione $p_{x}$')
axs[1].set_title('Componente velocità $v_{x}$')
axs[0].set_xlabel('Tempo [s]')
axs[1].set_xlabel('Tempo [s]')
axs[0].grid(True)
axs[1].grid(True)
plt.tight_layout()
plt.show()

# Plot errore di stima e controllo
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
for i in range(N):
    axs[0].plot(time[:251], error[:251, i])
    axs[1].plot(u_time[:700], ux[:700, i])
axs[0].set_title('Errore di stima $||x - \hat{x}||$')
axs[1].set_title('Controllo $u_x$')
axs[0].set_xlabel('Tempo [s]')
axs[1].set_xlabel('Tempo [s]')
axs[0].grid(True)
axs[1].grid(True)
plt.tight_layout()
plt.show()

# Plot 3D traiettorie
initial_pos = positions[0]
final_pos = positions[-1]
mean_initial = np.mean(initial_pos, axis=0)
colors = plt.cm.viridis(np.linspace(0, 1, N))

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
for i in range(N):
    ax.plot(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2], color=colors[i])
    ax.plot([initial_pos[i, 0]], [initial_pos[i, 1]], [initial_pos[i, 2]], 'o', color=colors[i], markersize=2)

for i in range(N):
    for j in [(i - 1) % N, (i + 1) % N]:
        ax.plot([initial_pos[i, 0], initial_pos[j, 0]],
                [initial_pos[i, 1], initial_pos[j, 1]],
                [initial_pos[i, 2], initial_pos[j, 2]],
                linestyle='dashed', color='gray', linewidth=0.2, alpha=0.6)

ax.plot([mean_initial[0]], [mean_initial[1]], [mean_initial[2]], 'r.', markersize=6, label='Centroid')

flat_pos = positions.reshape(-1, 3)
margin = 1
ax.set_xlim(flat_pos[:, 0].min() - margin, flat_pos[:, 0].max() + margin)
ax.set_ylim(flat_pos[:, 1].min() - margin, flat_pos[:, 1].max() + margin)
ax.set_zlim(flat_pos[:, 2].min() - margin, flat_pos[:, 2].max() + margin)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Tracciati 3D e topologia iniziale')
ax.legend()
plt.tight_layout()
plt.show()
