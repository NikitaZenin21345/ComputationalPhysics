import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Define constants
V = 1  # Example value for V, user can modify it
a = 1  # Example value for a, user can modify it

# Define k values (wavevector range)
k_values = np.linspace(-np.pi/a, np.pi/a, 1000)


def E_k_with_arccos(k_values, V, a):
    energies = []
    for k_val in k_values:
        # Define the function argument for arccos
        argument = lambda E: (V / (2 * np.sqrt(E))) * np.sin(np.sqrt(E) * a) + np.cos(np.sqrt(E) * a)

        # Guess energy values and filter those that are in the valid range for arccos
        energy_guess = np.linspace(0.01, 10, 1000)
        valid_E = [E for E in energy_guess if np.abs(argument(E) - np.cos(k_val * a)) < 1e-3]

        # Append the first valid solution or None if no solution is found
        if valid_E:
            energies.append(valid_E[0])
        else:
            energies.append(None)

    return np.array(energies, dtype=np.float64)


# Calculate energies using the modified approach
energies_arccos = E_k_with_arccos(k_values, V, a)

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(k_values, energies_arccos, label='E(k) with arccos', color='orange')
plt.title('Energy Dispersion Relation E(k) using arccos')
plt.xlabel('k (Wavevector)')
plt.ylabel('E (Energy)')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(True)
plt.legend()
plt.show()
