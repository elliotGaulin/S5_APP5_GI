import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# Paramètres
# -----------------------------
N = 100000
lambdas = [10, 50, 100]  # joueurs / minute

mu_semaine = 280.58  # temps moyen hebdomadaire en minutes
sigma_semaine = 50.38

# -----------------------------
# Échantillonnage
# -----------------------------
def sample_exponential_inverse(n, lambda_rate):
    u = np.random.uniform(0.0, 1.0, n)
    u = np.clip(u, np.finfo(float).tiny, 1.0)
    return -np.log(u) / lambda_rate

def sample_normal_inverse_cdf(n, mean, std):
    u = np.random.uniform(0.0, 1.0, n)
    u = np.clip(u, np.finfo(float).tiny, 1.0 - np.finfo(float).eps)
    z = stats.norm.ppf(u)
    return mean + std * z

# -----------------------------
# Simulation Monte-Carlo
# -----------------------------
def simulation(lambda_rate):
    # 1. P ~ Exp(lambda) via transformation inverse
    P = sample_exponential_inverse(N, lambda_rate)

    # 2. Instants d'arrivée
    t_arrival = np.cumsum(P)

    # 3. Q ~ N(mu, sigma) via inverse CDF
    Q = sample_normal_inverse_cdf(N, mu_semaine, sigma_semaine)
    Q = np.maximum(Q, 0)  # éviter les temps négatifs

    # 4. Instants de départ
    t_departure = t_arrival + Q

    # 5. Estimer le nombre moyen de joueurs actifs sur 100 instants
    t_start = t_arrival[int(N * 0.1)]
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
    plt.title("Histogramme Q (Normale, inverse CDF)")
    plt.xlabel("Temps de jeu (minutes/semaine)")
    plt.ylabel("Fréquence")
    plt.show()

    # Estimation théorique rapide
    moyenne_theorique = lam * mu_semaine
    print(f"Nombre moyen de joueurs (théorique) ≈ {moyenne_theorique:.2f}")