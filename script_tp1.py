import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# --- Constantes físicas ---
hbar = 1.054571817e-34   # J*s
m = 9.10938356e-31       # kg
eV = 1.602176634e-19     # J
h = 6.62607015e-34       # J*s (constante de Planck)

# --- Parámetros espaciales ---
Lx = 240e-9   # longitud del dominio [m]
dx = 0.05e-9
Nx =int(Lx/dx ) # número de puntos
x = np.linspace(0, Lx, Nx)
x_nm = x*1e9  # en nanómetros

# --- Parámetros temporales ---
dt_max = 0.15 * 2 * m * dx**2 / hbar
dt = 0.5*dt_max
Nt = int(500e-15/dt)

# Energía cinética media inicial solicitada
Ek0_eV = 0.05            # eV
Ek0 = Ek0_eV * eV        # J

# --- Condición inicial: paquete gaussiano ---
x0 = 110e-9
sigma = 4e-9

p0 = np.sqrt(2 * m * Ek0)
lambda0 = h/ p0
k0 = 2 * np.pi / lambda0

#--- Parametro ra 
ra = hbar * dt / (2*m*dx**2)

if ra >= 0.15:
    raise ValueError("Condición de estabilidad no cumplida: ra >= 0.15")


envelope = np.exp(-(x-x0)**2/(2*sigma**2))
psi_real = envelope * np.cos(k0*(x-x0))
psi_imag = envelope * np.sin(k0*(x-x0))

max_densidad = 1/(np.sqrt(np.pi)*sigma)

# Normalización inicial
norm = np.sqrt(np.trapz(psi_real**2 + psi_imag**2, x))
psi_real /= norm
psi_imag /= norm

# Listas para guardar valores esperados
x_means = []
p_means = []
Ek_means = []
Ep_means = []
E_totals = []
deltas_x = []
deltas_p = []
productos_deltas_normalizados = []

# --- Snapshots para guardar tiempos especificos ---
psi_sqr_snapshots = []
times = []

idx = np.argmin(np.abs(x - Lx/2))

############################ Potenciales  #########################################################

class Casos(Enum):
    PARTICULA_LIBRE = 0
    ESCALON_0025 = 1
    ESCALON_025  = 2
    BARRERA_1NM  = 3
    BARRERA_2NM  = 4
    BARRERA_10NM = 5

# Elegir un caso
caso = Casos.BARRERA_1NM

V = np.zeros(Nx) # comun a todos

if(caso.value == Casos.PARTICULA_LIBRE.value):
    #Solo Para condiciones iniciales
    target_times = [0]  # en femtosegundos
else: 
        # Para evoluciónes temporales
    target_times = [0, 30, 80, 450]  # en femtosegundos
tolerance = dt*1e15/2  # mitad de un paso temporal en fs, para calcular bien los valores esperados 

    
if(caso.value == Casos.ESCALON_0025.value):
    # Escalón de potencial 0.025 eV
    V[x < Lx/2] = 0*eV
    V[x >= Lx/2] =0.025 * eV  # 0.025 eV a partir de x = Lx/2


if(caso.value == Casos.ESCALON_025.value):
    # Escalón de potencial 0.025 eV
    V[x < Lx/2] = 0*eV
    V[x >= Lx/2] =0.025 * eV  # 0.025 eV a partir de x = Lx/2

     
if(caso.value >= Casos.BARRERA_1NM.value):
    # # # Potencial barrera
    Ebarr = 0.150 * eV  #convertir eV a Joules
    d = 1 # por defecto

    if(caso.value >= Casos.BARRERA_1NM.value):
        d_nm = 1
    if(caso.value >= Casos.BARRERA_2NM.value):
        d_nm = 2
    if(caso.value >= Casos.BARRERA_10NM.value):
        d_nm = 10
                
                
    d = d_nm * 1e-9     #convertir a metos

    V = np.zeros(Nx)    #inicializar el potencial
    x_start = Lx/2

     ##indices de la barrera
    idx_start = np.argmin(np.abs(x - x_start))
    idx_end   = np.argmin(np.abs(x - (x_start + d)))

    V[idx_start:idx_end] = Ebarr

    
##############################################################################################

plt.figure(figsize=(8,5))

if(caso.value == Casos.PARTICULA_LIBRE.value):
    #Componentes Real e imaginaria en tiempo 0 
    plt.plot(x*1e9, 2e-14*psi_real*max_densidad, label=fr"$\Psi Real, \; t = {0:.0f}\ \mathrm{{fs}}$", alpha=0.9, linewidth=3)
    plt.plot(x*1e9,2e-14*psi_imag*max_densidad, label=fr"$\Psi Imag, \; t = {0:.0f}\ \mathrm{{fs}}$", alpha=0.9, linewidth=3)

# --- Evolución temporal ---
for n in range(Nt):

    # Paso 1: actualizar parte real
    lap_imag = (np.roll(psi_imag, -1) - 2*psi_imag + np.roll(psi_imag, 1)) / dx**2
    psi_real_new = psi_real - (hbar*dt/(2*m)) * lap_imag + (dt/hbar)*V*psi_imag

    # Paso 2: actualizar parte imaginaria
    lap_real = (np.roll(psi_real_new, -1) - 2*psi_real_new + np.roll(psi_real_new, 1)) / dx**2
    psi_imag_new = psi_imag + (hbar*dt/(2*m)) * lap_real - (dt/hbar)*V*psi_real_new


    norm = np.sqrt(np.trapz(psi_real_new**2 + psi_imag_new**2, x))

    # Actualizar variables
    psi_real, psi_imag = psi_real_new/norm, psi_imag_new/norm


    # Calcular densidad de probabilidad
    prob_density = psi_real**2 + psi_imag**2
    current_time_fs = n*dt*1e15  # tiempo actual en femtosegundos

    # --- Valores esperados en los tiempos target ---
    for T in target_times:
        if abs(current_time_fs - T) < tolerance:
            psi = psi_real + 1j*psi_imag
            psi_conj = np.conjugate(psi)

            x_mean = np.sum(np.abs(psi)**2 * x) * dx
            x_mean_2 = np.sum(np.abs(psi)**2 * x**2) * dx
            grad_psi = (np.roll(psi, -1) - np.roll(psi, 1)) / (2*dx)
            p_mean = np.sum(psi_conj * (-1j*hbar) * grad_psi) * dx
        
            lap_psi = (np.roll(psi, -1) - 2*psi + np.roll(psi, 1)) / dx**2
            Ek_mean = np.real(np.sum(psi_conj * (-hbar**2/(2*m)) * lap_psi) * dx)
            Ep_mean = np.real(np.sum(psi_conj * V * psi) * dx)
            p_mean_2 = 2 * m * Ek_mean
            E_total = Ek_mean + Ep_mean


            # Incertidumbre de x: delta x = 2 * sqrt(<x^2> - <x>^2)
            delta_x = 2 * np.sqrt(x_mean_2 - x_mean**2)

            
            # Incertidumbre de p: delta p = 2 * sqrt(<p^2> - <p>^2)
            delta_p = 2 * np.sqrt(np.real(p_mean_2) - np.real(p_mean)**2)

            
            producto_deltas_normalizado = (delta_x*delta_p)/(hbar/2)
            
            # Guardar en las listas
            x_means.append(x_mean)
            p_means.append(p_mean)
            Ek_means.append(Ek_mean)
            Ep_means.append(Ep_mean)
            E_totals.append(E_total)
            deltas_x.append(delta_x)
            deltas_p.append(delta_p)
            productos_deltas_normalizados.append(producto_deltas_normalizado)

    # Guardar snapshots cada cierto tiempo
    if n % 400 == 0:
        psi_sqr_snapshots.append(prob_density.copy())
        times.append(n*dt*1e15)  # en femtosegundos
        #P_total = np.sum(prob_density)*dx
        P_total = np.trapz(prob_density,x)#toma promedio , es mas preciso 
        print(f"Paso {n}, Probabilidad total: {P_total:.6f}")


# --- Graficar snapshots en tiempos específicos ---

tolerance = 0.5  # tolerancia en fs para encontrar el snapshot más cercano
for psi_sqr, t in zip(psi_sqr_snapshots, times):
#  revisar si t está cerca de alguno de los tiempos objetivo
    if any(abs(t - T) < tolerance for T in target_times):
        plt.plot(x*1e9, Ek0_eV*(psi_sqr/max_densidad), label=fr"$|\Psi|^2, \; t = {t:.0f}\ \mathrm{{fs}}$", alpha=0.9, linewidth=3)



if(caso.value >= Casos.BARRERA_1NM.value):
    #caso barreras   
    plt.plot(x*1e9,(V/eV),linestyle = "--", label=f"V(x = Lx/2) = {V[idx]/eV:.3f} eV , d = {d_nm} nm" , color ="gray", linewidth = 3 )
else :
    #Casos Escalones y Particula libre 
    plt.plot(x*1e9,(V/eV),linestyle = "--", label=f"V(x = Lx/2) = {V[idx]/eV:.3f} eV " , color ="gray", linewidth = 3 )


    
for T, x_val, p_val, Ek_val, Ep_val, E_val, delta_x, delta_p, producto_deltas_normalizado in zip(target_times, x_means, p_means, Ek_means, Ep_means, E_totals, deltas_x,deltas_p,productos_deltas_normalizados):
    print(f"t = {T} fs: <x>={x_val:.3e} m, <p>={p_val:.3e} kg·m/s, <Ek>={Ek_val/eV:.3f} eV, <Ep>={Ep_val/eV:.3f} eV, <E>={E_val/eV:.3f} eV ", fr"$\delta x = {delta_x}$", fr"$\delta p = {delta_p}$",fr"$deltas/(hbar/2) = {producto_deltas_normalizado}$" )


textstr = (
    f'P_total = {P_total:.4f}\n'
    f'ra = {ra:.4f}\n'
    f'Δt = {dt:.2e} s\n'
    f'Δx = {dx:.2e} m\n'

)

# Coordenadas relativas del cuadro (x, y) en fracción de la figura (0 a 1)
plt.gca().text(
    0.025, 0.95, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)



def coeficientes_escalon(E, V0, m, hbar):
    if E > V0:
        k1 = np.sqrt(2*m*E) / hbar
        k2 = np.sqrt(2*m*(E - V0)) / hbar
        R = ((k1 - k2) / (k1 + k2))**2
        T = (4*k1*k2) / (k1 + k2)**2
    else:
        # caso E < V0 → no hay transmisión propagante
        R = 1.0
        T = 0.0
    return R, T

def coeficientes_barrera(E, V0, d, m, hbar):
    if E > V0:
        k1 = np.sqrt(2*m*E) / hbar
        k2 = np.sqrt(2*m*(E - V0)) / hbar
        T = 1.0 / (1.0 + (V0**2 / (4*E*(E - V0))) * np.sin(k2*d)**2)
    else:
        k1 = np.sqrt(2*m*E) / hbar
        kappa = np.sqrt(2*m*(V0 - E)) / hbar
        T = 1.0 / (1.0 + (V0**2 / (4*E*(V0 - E))) * np.sinh(kappa*d)**2)

    R = 1.0 - T
    return R, T


V0 = 0.15*eV



if(caso.value >= Casos.BARRERA_1NM.value):
    R, T = coeficientes_barrera(Ek0, V0,d_nm, m, hbar)

else:
    R, T = coeficientes_escalon(Ek0, V0, m, hbar)

print(f"Coeficientes para escalón V = {V0/eV:.3f} eV:")
print(f"  Reflexión R = {R:.3f}")
print(f"  Transmisión T = {T:.3f}")
print(f"  Verificación: R + T = {R+T:.3f}")



def distancia_penetracion(E, V0, m, hbar):
    if E < V0:
        kappa = np.sqrt(2*m*(V0 - E)) / hbar
        delta = 1 / kappa
        return delta
    else:
        return None  # no hay decaimiento si E > V0

delta = distancia_penetracion(Ek0, V0, m, hbar)

if delta is not None:
    print(f"Distancia de penetración δ = {delta*1e9:.2f} nm")


plt.xlabel("x (nm)")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.show()
