import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Paramètres
# -----------------------------
N = 10000  # nombre de joueurs (au moins 10 000)
lambdas = [10, 50, 100]  # joueurs / minute

mu = 280.58 / 7
sigma = 50.38 / 7

# -----------------------------
# Échantillonnage
# -----------------------------
def sample_exponential_inverse(n, lambda_rate):
    u = np.random.uniform(0.0, 1.0, n)
    u = np.clip(u, np.finfo(float).tiny, 1.0)
    return -np.log(u) / lambda_rate

def sample_normal_box_muller(n, mean, std):
    u1 = np.random.uniform(0, 1, n)
    u2 = np.random.uniform(0, 1, n)
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mean + std * z



def simulation(lambda_rate):
    # 1. P ~ Exp(lambda) via transformation inverse
    P = sample_exponential_inverse(N, lambda_rate)

    # 2. Instants d'arrivée
    t_arrival = np.cumsum(P)

    # 3. Q ~ N(mu, sigma) via NumPy
    Q = sample_normal_box_muller(N, mu, sigma)
    Q = np.maximum(Q, 0)  # éviter les temps de jeu négatifs

    # 4. Instants de départ
    t_departure = t_arrival + Q

    # 5. Estimer le nombre moyen de joueurs actifs sur 100 instants
    t_start = t_arrival[int(N * 0.1)]  # éviter les premiers instants pour une meilleure estimation
    t_end = t_arrival[-1]
    T_values = np.linspace(t_start, t_end, 100)

    active_counts = []
    for T in T_values:
        active = np.sum((t_arrival <= T) & (t_departure >= T))
        active_counts.append(active)

    moyenne = np.mean(active_counts)

    return P, Q, moyenne

# -----------------------------
# Exécution
# -----------------------------
for lam in lambdas:
    P, Q, moyenne = simulation(lam)

    print(f"\nλ = {lam} joueurs/min")
    print(f"Nombre moyen de joueurs connectés ≈ {moyenne:.2f}")

    # Histogramme P (exponentielle)
    plt.figure()
    plt.hist(P, bins=50, density=True)
    plt.title(f"Histogramme P (Exponentielle) - λ={lam}")
    plt.xlabel("Temps entre arrivées")
    plt.ylabel("Fréquence")
    plt.show()

    # Histogramme Q (normale)
    plt.figure()
    plt.hist(Q, bins=50, density=True)
    plt.title("Histogramme Q (Normale, Box-Muller)")
    plt.xlabel("Temps de jeu")
    plt.ylabel("Fréquence")
    plt.show()