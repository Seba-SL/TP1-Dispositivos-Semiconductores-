import numpy as np
import matplotlib.pyplot as plt

# --- Constantes físicas ---
hbar = 1.054571817e-34   # J*s
m = 9.10938356e-31       # kg
eV = 1.602176634e-19     # J
h = 6.62607015e-34       # J*s (constante de Planck)

# --- Parámetros espaciales ---
Lx = 240e-9   # longitud del dominio [m]
Nx = 1200    # número de puntos
dx = 0.05e-9
x = np.linspace(0, Lx, Nx)
x_nm = x*1e9  # en nanómetros

Nt = 180000
dt_max = 0.15 * 2 * m * dx**2 / hbar
dt = 0.5*dt_max


# --- Potencial: escalón en x = 20 nm ---
V = np.zeros(Nx)
V[x_nm >= Lx/2 ] = 0.015*eV   # 0.015 eV

# --- Condición inicial: paquete gaussiano ---
x0 = 110e-9
sigma = 4e-9

# Energía cinética media inicial solicitada
Ek0_eV = 0.05            # eV
Ek0 = Ek0_eV * eV        # J


ra = hbar * dt / (2*m*dx**2)

if ra >= 0.15:
    raise ValueError("Condición de estabilidad no cumplida: ra >= 0.15")



p0 = np.sqrt(2 * m * Ek0)
lambda0 = h/ p0
k0 = 2 * np.pi / lambda0


envelope = np.exp(-(x-x0)**2/(2*sigma**2))
psi_real = envelope * np.cos(k0*(x-x0))
psi_imag = envelope * np.sin(k0*(x-x0))

# Normalización inicial
norm = np.sqrt(np.trapz(psi_real**2 + psi_imag**2, x))
psi_real /= norm
psi_imag /= norm

# --- Potencial (libre en este caso) ---
V = np.zeros_like(x)

# --- Snapshots ---
psi_sqr_snapshots = []
times = []

# --- Evolución temporal ---
for n in range(Nt):

    # Paso 1: actualizar parte real
    lap_imag = (np.roll(psi_imag, -1) - 2*psi_imag + np.roll(psi_imag, 1)) / dx**2
    psi_real_new = psi_real - (hbar*dt/(2*m)) * lap_imag + (dt/hbar)*V*psi_imag

    # Paso 2: actualizar parte imaginaria
    lap_real = (np.roll(psi_real_new, -1) - 2*psi_real_new + np.roll(psi_real_new, 1)) / dx**2
    psi_imag_new = psi_imag + (hbar*dt/(2*m)) * lap_real - (dt/hbar)*V*psi_real_new

    # Actualizar variables
    psi_real, psi_imag = psi_real_new, psi_imag_new

    # Calcular densidad de probabilidad
    prob_density = psi_real**2 + psi_imag**2

    # Guardar snapshots cada cierto tiempo
    if n % 500 == 0:
        psi_sqr_snapshots.append(prob_density.copy())
        times.append(n*dt*1e15)  # en femtosegundos
        P_total = np.trapz(prob_density, x)
        print(f"Paso {n}, Probabilidad total: {P_total:.6f}")



# --- Cálculo de valores esperados (en el último estado) ---

# función de onda compleja Ψ = Re + i Im
psi = psi_real + 1j * psi_imag
psi_conj = np.conjugate(psi)

# 1) Valor esperado de la posición <x>
x_mean = np.sum((np.abs(psi)**2) * x) * dx

# 2) Valor esperado del momento <p>
#   Usamos diferencias centrales: (ψ[n+1] - ψ[n-1]) / (2dx)
grad_psi = (np.roll(psi, -1) - np.roll(psi, 1)) / (2*dx)
p_mean = np.sum(psi_conj * (-1j * hbar) * grad_psi) * dx

# 3) Valor esperado de energía cinética <Ek>
lap_psi = (np.roll(psi, -1) - 2*psi + np.roll(psi, 1)) / dx**2
Ek_mean = np.sum(psi_conj * (- (hbar**2) / (2*m)) * lap_psi) * dx
Ek_mean = np.real(Ek_mean)  # debe ser real

# 4) Valor esperado de energía potencial <Ep>
Ep_mean = np.sum(psi_conj * V * psi) * dx
Ep_mean = np.real(Ep_mean)

# 5) Energía total <E>
E_total = Ek_mean + Ep_mean

print("\n--- Valores esperados (último estado) ---")
print(f"<x>   = {x_mean:.3e} m")
print(f"<p>   = {p_mean:.3e} kg·m/s")
print(f"<Ek>  = {Ek_mean/eV:.3f} eV")
print(f"<Ep>  = {Ep_mean/eV:.3f} eV")
print(f"<E>   = {E_total/eV:.3f} eV")



# --- Graficar snapshots ---
for psi_sqr, t in zip(psi_sqr_snapshots, times):
    plt.plot(x*1e9, psi_sqr, label=f"t = {t:.1f} fs",alpha = 0.9, linewidth = 3)

textstr = (
    f'P_total = {P_total:.4f}\n'
    f'ra = {ra:.4f}\n'
    f'Δt = {dt:.2e} s\n'
    f'Δx = {dx:.2e} m\n'
    f'\n--- Valores esperados (último estado) ---\n'
    f'<x>   = {x_mean:.3e} m\n'
    f'<p>   = {p_mean:.3e} kg·m/s\n'
    f'<Ek>  = {Ek_mean/eV:.3f} eV\n'
    f'<Ep>  = {Ep_mean/eV:.3f} eV\n'
    f'<E>   = {E_total/eV:.3f} eV\n'

)

# Coordenadas relativas del cuadro (x, y) en fracción de la figura (0 a 1)
plt.gca().text(
    0.1, 0.95, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.xlabel("x (nm)")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.show()

# # --- Evolución temporal ---
# snapshot_times_fs = [0, 30, 80, 450]  # tiempos deseados en fs
# psi_sqr_snapshots = []
# times = []

# for n in range(Nt):
#     # Laplaciano
#     lap_real = (np.roll(psi_real,-1) - 2*psi_real + np.roll(psi_real,1)) / dx**2
#     lap_imag = (np.roll(psi_imag,-1) - 2*psi_imag + np.roll(psi_imag,1)) / dx**2

#     # Ecuación de Schrödinger explícita
#     psi_real_new = psi_real + dt*( -(hbar/(2*m))*lap_imag + (V/hbar)*psi_imag )
#     psi_imag_new = psi_imag + dt*(  (hbar/(2*m))*lap_real  - (V/hbar)*psi_real )

#     psi_real, psi_imag = psi_real_new, psi_imag_new

#     # tiempo actual en fs
#     t_fs = n * dt * 1e15

#     # Guardar solo los snapshots más cercanos a los tiempos deseados
#     for t_target in snapshot_times_fs:
#         if len(times) < len(snapshot_times_fs) and abs(t_fs - t_target) < dt*1e15:
#             psi_sqr = psi_real**2 + psi_imag**2
#             psi_sqr_snapshots.append(psi_sqr.copy())
#             times.append(t_fs)
#             # Normalización opcional
#             norm = np.sqrt(np.trapz(psi_real**2 + psi_imag**2, x))
#             psi_real /= norm
#             psi_imag /= norm
#             break  # evita duplicados

# # --- Gráfico ---
# fig, ax1 = plt.subplots()
# for i, psi_sqr in enumerate(psi_sqr_snapshots):
#     ax1.plot(x_nm, psi_sqr, label=f"t = {times[i]:.1f} fs")

# ax1.set_xlabel("x [nm]")
# ax1.set_ylabel(r"$|\psi(x,t)|^2$")
# ax1.legend()

# # Potencial
# ax2 = ax1.twinx()
# ax2.set_ylabel("Potencial [eV]")
# plt.title("Evolución Temporal Paquete de Ondas")
# plt.show()
