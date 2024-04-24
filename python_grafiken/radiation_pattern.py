#%%
import numpy as np
import matplotlib.pyplot as plt

# Funktion zur Berechnung des Strahlungsmusters
def radiation_pattern(theta):
    return np.abs(np.cos(np.pi * np.cos(theta) / 2))

# Erstellen eines Gitters von Winkeln theta und phi für den 3D-Plot
theta, phi = np.meshgrid(np.linspace(0, np.pi, 100), np.linspace(0, 2 * np.pi, 100))

# Berechnung des Strahlungsmusters für den 3D-Plot
R_3d = radiation_pattern(theta)

# Erstellung der Figure für den 3D-Plot
fig_3d = plt.figure(figsize=(10, 8))
ax1 = fig_3d.add_subplot(projection='3d')

# 3D-Plot des Strahlungsmusters
ax1.plot_wireframe(R_3d * np.sin(theta) * np.cos(phi), R_3d * np.sin(theta) * np.sin(phi), R_3d * np.cos(theta), 
                 rstride=1, cstride=1, cmap='viridis', alpha=0.3, linewidth=0.5, antialiased=True, zorder=1)

# Antenne als dicken Stab entlang der Z-Achse visualisieren
antenna_length = 1  # Stellt die Länge der Antenne dar
antenna_radius = 0.05  # Radius für die Darstellung der Dicke der Antenne

ax1.plot3D([0, 0], [0, 0], [-antenna_length/2, antenna_length/2], color='k', linewidth=2*antenna_radius*100, alpha=0.7, linestyle='-', solid_capstyle='round', zorder=10)
ax1.text(0, 0.2, antenna_length/2+0.01, r'Antenne', color='black', fontsize=20, ha='center')



ax1.set_title(r'3D Antennendiagramm mit Antenne', pad=20, fontsize=15)
ax1.set_xlabel('X', fontsize=20)
ax1.set_ylabel('Y', fontsize=20)
ax1.set_zlabel('Z', fontsize=20)

plt.tight_layout()
plt.savefig('Versuch_Arduino_Skript/images/3d_radiation_pattern.pdf')

#%%

# Winkelbereiche für Polarkoordinaten-Plots
theta_range = np.linspace(0, 2 * np.pi, 1000)
phi_range = np.linspace(0, 2 * np.pi, 1000)

# Strahlungsmuster für die Ebenen
R_xy = radiation_pattern(np.pi / 2)
R_xz = radiation_pattern(theta_range)
R_yz = radiation_pattern(theta_range)

# Plot-Einstellungen für Polarkoordinaten
fig_polar, axes = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 8))

# XY-Ebene in Polarkoordinaten
# rgb value for plot color 
axes[0].plot(phi_range, np.full_like(phi_range, R_xy), color=(0.204, 0.373, 0.667))
axes[0].set_title(r'XY-Ebene', pad=20, fontsize=15)
axes[0].set_theta_zero_location('N')
axes[0].set_theta_direction(-1)
axes[0].set_xlabel(r'$\phi$')  # Greek letter Phi in LaTeX math format

# XZ-Ebene in Polarkoordinaten
axes[1].plot(theta_range, R_xz, color=(0.204, 0.373, 0.667))
axes[1].set_title(r'XZ-Ebene', pad=20, fontsize=15)
axes[1].set_theta_zero_location('N')
axes[1].set_theta_direction(-1)
axes[1].set_xlabel(r'$\theta$')  # Greek letter Theta in LaTeX math format

# YZ-Ebene in Polarkoordinaten
axes[2].plot(theta_range, R_yz, color=(0.204, 0.373, 0.667))
axes[2].set_title(r'YZ-Ebene', pad=20, fontsize=15)
axes[2].set_theta_zero_location('N')
axes[2].set_theta_direction(-1)
axes[2].set_xlabel(r'$\theta$')  # Greek letter Theta in LaTeX math format

plt.tight_layout()
plt.savefig('Versuch_Arduino_Skript/images/polar_radiation_pattern.pdf')

#%%
# Gleiche 2d Diagramme wie vorher aber ohne plot, zum selbst einzeichnen

# Winkelbereiche für Polarkoordinaten-Plots
theta_range = np.linspace(0, 2 * np.pi, 1000)
phi_range = np.linspace(0, 2 * np.pi, 1000)

# Strahlungsmuster für die Ebenen
R_xy = radiation_pattern(np.pi / 2)
R_xz = radiation_pattern(theta_range)
R_yz = radiation_pattern(theta_range)

# Plot-Einstellungen für Polarkoordinaten
fig_polar, axes = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 8))

# XY-Ebene in Polarkoordinaten
# rgb value for plot color
# axes[0].plot(phi_range, np.full_like(phi_range, R_xy), color=(0.204, 0.373, 0.667))
axes[0].set_title(r'XY-Ebene', pad=20, fontsize=15)
axes[0].set_theta_zero_location('N')
axes[0].set_theta_direction(-1)
axes[0].set_xlabel(r'$\phi$')  # Greek letter Phi in LaTeX math format

# XZ-Ebene in Polarkoordinaten
# axes[1].plot(theta_range, R_xz, color=(0.204, 0.373, 0.667))
axes[1].set_title(r'XZ-Ebene', pad=20, fontsize=15)
axes[1].set_theta_zero_location('N')
axes[1].set_theta_direction(-1)
axes[1].set_xlabel(r'$\theta$')  # Greek letter Theta in LaTeX math format

# YZ-Ebene in Polarkoordinaten
# axes[2].plot(theta_range, R_yz, color=(0.204, 0.373, 0.667))
axes[2].set_title(r'YZ-Ebene', pad=20, fontsize=15)
axes[2].set_theta_zero_location('N')
axes[2].set_theta_direction(-1)
axes[2].set_xlabel(r'$\theta$')  # Greek letter Theta in LaTeX math format

plt.tight_layout()
plt.savefig('Versuch_Arduino_Skript/images/polar_radiation_pattern_empty.pdf')


# %%
