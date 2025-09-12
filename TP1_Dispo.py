import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation

# --- Constantes físicas ---

# hbar → constante de Planck reducida.

# m → masa del electrón.

# eV → conversión de electronvoltios a joules.

hbar = 1.054571817e-34   # J*s
m = 9.10938356e-31       # kg
eV = 1.602176634e-19     # J

# --- Parámetros numéricos ---

# Nx → discretización espacial (malla de 5000 puntos).

# Lx → tamaño total del dominio en metros.

# dx → paso espacial (distancia entre puntos de la malla).

# dt → paso temporal calculado para cumplir condición de estabilidad del método explícito.

# Nt → número de pasos de tiempo.


Lx = 240e-9
dx = 0.05e-9

dt_max = 0.15 * 2 * m * dx**2 / hbar
dt = 0.5*dt_max


Nx = int(Lx/dx) + 1

print('Nx :' , {Nx})

t_max = 500*10**-15

Nt = 500#int(t_max/dt) + 1 

print('\nNt :' , {Nt})

# Guardar snapshots cada Nskip pasos
Nskip = max(1, Nt // 500)  # ajustá 100 según la cantidad de snapshots que quieras


ra = hbar*dt/(2*m*dx**2)

# Condición de estabilidad
# ra es un parámetro que asegura que el método explícito no se vuelva inestable (similar al CFL en ondas clásicas).

ra = hbar * dt / (2*m*dx**2)
print("ra =", ra)

print("dt =", dt)
if ra >= 0.15:
    raise ValueError("Condición de estabilidad no cumplida: ra >= 0.15")

# --- Malla espacial ---
# Genera un vector de posiciones equiespaciadas desde 0 hasta Lx.
x = np.linspace(0, Lx, Nx)


# --- Potencial ---
# Se simula un electrón libre, ya que el potencial es cero en todo el dominio.
# (partı́cula sometida a diferentes potenciales que no tienen la capacidad de ligar a la partı́cula)

V = np.zeros_like(x)

# --- Condición inicial ---
x0 = 110e-9
sigma = 4e-9
l_onda = 5.45e-9
k0 = 2*np.pi/l_onda

envelope = np.exp(-(x-x0)**2/(sigma**2))
psi_real = envelope * np.cos(k0*(x-x0))
psi_imag = envelope * np.sin(k0*(x-x0))

# Se genera un paquete de onda gaussiano centrado en x0.

# sigma → ancho del paquete.

# k0 → número de onda (momento inicial del electrón).

# La función de onda se separa en parte real e imaginaria para poder integrar explícitamente.

# Normalización

# Calcula la integral de la densidad de probabilidad y normaliza la función de onda para que:

norm = np.sqrt(np.trapz(psi_real**2 + psi_imag**2, x))
psi_real /= norm
psi_imag /= norm

# --- Constantes ---
# simplifican la ecuación de Schrödinger discreta.
C1 = hbar / (2*m)
C2 = 1.0/hbar

# --- Guardamos datos para plotear después ---
data_to_plot = []

# --- Evolución temporal ---
# Calcula la evolución temporal de la función de onda usando diferencias finitas:

# psi_real depende del laplaciano de psi_imag y del potencial.

# psi_imag depende del laplaciano de psi_real y del potencial.

# Esto es la discretización explícita de Schrödinger.




# n recorre los pasos de tiempo, de 0 hasta Nt-1.

# psiR_old y psiI_old son copias de la función de onda en el paso anterior, necesarias porque la actualización explícita requiere los valores antiguos.

# Cada iteración de este ciclo representa un incremento temporal de dt segundos.

# Condiciones de contorno
psi_real[0] = psi_real[-1] = 0.0
psi_imag[0] = psi_imag[-1] = 0.0

for n in range(Nt):
    psiR_old = psi_real.copy()
    psiI_old = psi_imag.copy()

    for i in range(1, Nx-1):
        
        psi_real[i] = psiR_old[i] + dt * ( C1 * (psiI_old[i+1] - 2*psiI_old[i] + psiI_old[i-1]) / dx**2 + C2 * V[i] * psiI_old[i] )
        psi_imag[i] = psiI_old[i] - dt * ( C1 * (psiR_old[i+1] - 2*psiR_old[i] + psiR_old[i-1]) / dx**2 + C2 * V[i] * psiR_old[i] )
       
    # Condiciones de contorno
    psi_real[0] = psi_real[-1] = 0.0
    psi_imag[0] = psi_imag[-1] = 0.0

    # --- Renormalización en cada paso temporal ---
    norm = np.sqrt(np.trapz(psi_real**2 + psi_imag**2, x))
    psi_real /= norm
    psi_imag /= norm

   # Guardar snapshots
    if n % Nskip == 0 or n == Nt-1:
        prob_density = psi_real**2 + psi_imag**2
        data_to_plot.append((n*dt, psi_real.copy(), psi_imag.copy(), prob_density.copy()))

# --- Ploteo ---

P_total = np.sum(prob_density)* dx  # suma discreta ≈ integral
print(f"Probabilidad total: {P_total:.4f}")  # debería estar cerca de 1 si está normalizado


# --- Ploteo ---
plt.figure(figsize=(12,6))
plt.plot(x*1e9, 5000*psi_real, color='blue', alpha=0.6, label='Re(Ψ) × 5000, para t = 0 ',linewidth = 3 )
plt.plot(x*1e9, 5000*psi_imag, color='red', alpha=0.6, label='Im(Ψ) × 5000, para t = 0',linewidth = 3 )
plt.plot(x*1e9, prob_density, color='green', alpha=0.8, label=f'|Ψ|², para t = 0',linewidth = 3 )
# --- Cuadro con parámetros ---
textstr = (
    f'P_total = {P_total:.4f}\n'
    f'ra = {ra:.4f}\n'
    f'Δt = {dt:.2e} s\n'
    f'Δx = {dx:.2e} m'
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
plt.ylabel("Amplitud / Densidad")
plt.title("Evolución de un paquete de onda en 1D (electrón)")
plt.legend()
plt.show()


## Graficos de los tiempos :


print(f"Número de snapshots guardados: {len(data_to_plot)}")
print("Tiempos guardados (fs):")
for t_snapshot, _, _, _ in data_to_plot:
    print(f"{t_snapshot*1e15:.2f} fs")

#los tiempos que queremos graficar
times_to_plot = [0, 30e-15, 70e-15, 440e-15]

#contamos cuántos subplots necesitamos
n_plots = len(times_to_plot)

fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots), sharex=True)

for ax, t_target in zip(axes, times_to_plot):
    #buscamos el snapshot más cercano a t_target
    closest_snapshot = min(data_to_plot, key=lambda s: abs(s[0] - t_target))
    t_snapshot, psiR, psiI, prob = closest_snapshot
    
    ax.plot(x*1e9, 5000*psiR, color='blue', alpha=0.6, label='Re(Ψ) × 5000')
    ax.plot(x*1e9, 5000*psiI, color='red', alpha=0.6, label='Im(Ψ) × 5000')
    ax.plot(x*1e9, prob, color='green', alpha=0.8, label='|Ψ|²')
 
    ax.set_ylabel("Amplitud / Densidad")
    ax.set_title(f"t = {t_snapshot*1e15:.0f} fs")
    ax.legend(loc='upper right')

axes[-1].set_xlabel("x (nm)")

plt.tight_layout()
plt.show()

