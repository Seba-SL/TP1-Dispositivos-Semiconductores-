import numpy as np
import matplotlib.pyplot as plt

# --- Constantes físicas ---
hbar = 1.054571817e-34   # J*s
m = 9.10938356e-31       # kg
eV = 1.602176634e-19     # J
h = 6.62607015e-34       # J*s

# --- Parámetros espaciales ---
Lx = 240e-9   # longitud del dominio [m]
dx = 0.05e-9
Nx = int(Lx/dx)
x = np.linspace(0, Lx, Nx)
x_nm = x*1e9  # en nanómetros

dt_max = 0.15 * 2 * m * dx**2 / hbar
dt = 0.5 * dt_max
Nt = int(500e-15/dt)

# --- Condición inicial: paquete gaussiano ---
x0 = 110e-9
sigma = 4e-9
Ek0_eV = 0.05
Ek0 = Ek0_eV * eV
p0 = np.sqrt(2 * m * Ek0)
lambda0 = h/p0
k0 = 2 * np.pi / lambda0
envelope = np.exp(-(x-x0)**2/(2*sigma**2))

max_densidad = 1/(np.sqrt(np.pi)*sigma)

def inicializar_paquete():
    psi_real = envelope * np.cos(k0*(x-x0))
    psi_imag = envelope * np.sin(k0*(x-x0))
    norm = np.sqrt(np.trapz(psi_real**2 + psi_imag**2, x))
    return psi_real/norm, psi_imag/norm

def evolucionar(psi_real, psi_imag, V, t_target_fs):
    """ Evoluciona hasta un tiempo específico y devuelve densidad |psi|^2 """
    Nt_target = int((t_target_fs*1e-15)/dt)
    for n in range(Nt_target):
        lap_imag = (np.roll(psi_imag,-1)-2*psi_imag+np.roll(psi_imag,1))/dx**2
        psi_real_new = psi_real - (hbar*dt/(2*m))*lap_imag + (dt/hbar)*V*psi_imag
        lap_real = (np.roll(psi_real_new,-1)-2*psi_real_new+np.roll(psi_real_new,1))/dx**2
        psi_imag_new = psi_imag + (hbar*dt/(2*m))*lap_real - (dt/hbar)*V*psi_real_new
        norm = np.sqrt(np.trapz(psi_real_new**2+psi_imag_new**2, x))
        psi_real, psi_imag = psi_real_new/norm, psi_imag_new/norm
        print("Paso ",{n})
    return psi_real**2 + psi_imag**2

# --- Definir casos ---
casos = {
    "[V = 0]": np.zeros(Nx),
    "[Escalón 0.025 eV]": np.where(x>=Lx/2, 0.025*eV, 0),
    "[Escalón 0.25 eV]": np.where(x>=Lx/2, 0.25*eV, 0),
    "[Barrera d=1 nm]": None,
    "[Barrera d=2 nm]": None,
    "[Barrera d=10 nm]": None
}

# Barreras
Ebarr = 0.150 * eV

idx = np.argmin(np.abs(x - Lx/2))


for d_nm in [1,2,10]:
    V = np.zeros(Nx)
    x_start = Lx/2
    idx_start = np.argmin(np.abs(x - x_start))
    idx_end   = np.argmin(np.abs(x - (x_start + d_nm*1e-9)))
    V[idx_start:idx_end] = Ebarr
    casos[f"[Barrera d={d_nm} nm]"] = V

# --- Simular todos los casos ---
t_target = 450  # fs
plt.figure(figsize=(8,10))

for i,(nombre,V) in enumerate(casos.items()):
    psi_real, psi_imag = inicializar_paquete()
    densidad = evolucionar(psi_real, psi_imag, V, t_target)
    plt.subplot(len(casos),1,i+1)
    plt.plot(x*1e9, Ek0_eV*(densidad/max_densidad), color ="red", alpha = 0.9, label=f"{nombre}, t={t_target} fs", linewidth = 3.5 )
 
    plt.plot(x*1e9,(V/eV),linestyle = "--", label=f"V(x = Lx/2) = {V[idx]/eV:.3f} eV" , color ="gray", linewidth = 3)
  
    plt.ylabel(r"$|\Psi|^2$")
    
    plt.ylim(0, 0.0165) 
    plt.legend(loc="upper right", fontsize=8)

plt.xlabel("x (nm)")
plt.tight_layout()
plt.show()
