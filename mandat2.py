import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2

# Load data from file
data_file = Path(__file__).resolve().parent / 'TempsDeJeu.txt'
with open(data_file, 'r') as f:
    data = [float(line.strip()) for line in f.readlines()]

data = np.array(data)

# ============== PART I: DESCRIPTIVE STATISTICS ==============
print("=" * 70)
print("STATISTIQUES DESCRIPTIVES DU TEMPS DE JEU")
print("=" * 70)

# Calculate statistics using unbiased estimators
mean = statistics.mean(data)  # Unbiased sample mean
median = statistics.median(data)
mode = statistics.mode(data)
variance = statistics.variance(data)  # Unbiased variance (using ddof=1)
std_dev = statistics.stdev(data)  # Unbiased standard deviation
minimum = np.min(data)
maximum = np.max(data)
range_val = maximum - minimum

# Create results table
results = {
    'Statistique': ['Moyenne', 'Médiane', 'Mode', 'Variance', 'Écart-type', 'Minimum', 'Maximum', 'Étendue'],
    'Valeur': [mean, median, mode, variance, std_dev, minimum, maximum, range_val]
}

results_df = pd.DataFrame(results)
print("\nTableau des Statistiques Descriptives:")
print(results_df.to_string(index=False))
print()

# ============== PART II: FREQUENCY TABLE AND HISTOGRAM ==============
print("=" * 70)
print("CRÉATION DE L'HISTOGRAMME ET TABLEAU DE FRÉQUENCES")
print("=" * 70)

# Determine number of classes using Sturges' rule
n = len(data)
num_classes = int(np.ceil(1 + np.log2(n)))
bins = np.linspace(minimum, maximum, num_classes + 1)

print(f"\nNombre d'observations: {n}")
print(f"Nombre de classes (Règle de Sturges): {num_classes}")
print("Justification: règle classique pour déterminer automatiquement le nombre de classes.")
print("Formule: k = ceil(1 + log2(n))")
frequencies, bin_edges = np.histogram(data, bins=bins)

# Create frequency table
classes = []
limits = []
centers = []
freq_relative = []
freq_cumulative = []

cumulative = 0
for i in range(len(frequencies)):
    lower = bin_edges[i]
    upper = bin_edges[i + 1]
    center = (lower + upper) / 2
    rel_freq = frequencies[i] / n
    cumulative += frequencies[i]
    
    classes.append(f"[{lower:.1f}, {upper:.1f}[")
    limits.append(f"[{lower:.1f}, {upper:.1f}[")
    centers.append(center)
    freq_relative.append(rel_freq)
    freq_cumulative.append(cumulative)

# Create DataFrame for frequency table
freq_table = pd.DataFrame({
    'Classes': classes,
    'Limites': limits,
    'Centres': [f"{c:.1f}" for c in centers],
    'Fréquences': frequencies,
    'Fréquences relatives': [f"{f:.4f}" for f in freq_relative],
    'Fréquences cumulées': freq_cumulative
})

print("\nTableau de Fréquences:")
print(freq_table.to_string(index=False))
print()

# ============== CREATE HISTOGRAM ==============
fig, ax = plt.subplots(figsize=(12, 6))

# Plot histogram with specified bins and classes
bars = ax.bar(centers, frequencies, width=(maximum - minimum) / num_classes, 
              edgecolor='black', alpha=0.7, color='steelblue')

ax.set_xlabel('Temps de jeu (minutes)', fontsize=12)
ax.set_ylabel('Fréquence', fontsize=12)
ax.set_title('Histogramme des Temps de Jeu des Joueurs', fontsize=14, fontweight='bold')
ax.set_xticks([c for c in centers])
ax.set_xticklabels([f"{c:.0f}" for c in centers], rotation=45)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('histogramme_temps_de_jeu.png', dpi=300, bbox_inches='tight')
print("Histogramme sauvegardé sous: histogramme_temps_de_jeu.png")
plt.show()

# ============== PART III: NORMALITY TEST ==============
print("=" * 70)
print("iii) TEST DE NORMALITÉ DES DONNÉES")
print("=" * 70)

print("\n1. OBSERVATION VISUELLE DE L'HISTOGRAMME:")
print("   L'histogramme a été créé précédemment (histogramme_temps_de_jeu.png)")
print("   Observez si la distribution ressemble à une cloche (courbe normale)")

alpha_test = 0.05
print("\n2. TEST DE KHI-DEUX (CHI-SQUARE) - TEST DE PEARSON:")
print("   (Test d'ajustement: compare données observées vs distribution théorique normale)")

# Perform chi-square goodness-of-fit test
# Expected frequencies for a normal distribution
expected_freq = []
for i in range(len(frequencies)):
    lower = bin_edges[i]
    upper = bin_edges[i + 1]
    # Probability in this bin under normal distribution
    prob = stats.norm.cdf(upper, loc=mean, scale=std_dev) - stats.norm.cdf(lower, loc=mean, scale=std_dev)
    expected_freq.append(prob * n)

expected_freq = np.array(expected_freq)

# Calculate chi-square statistic
# Merge adjacent classes until expected frequency >= 5 (Pearson requirement)
merged_classes = []
current = None

for i in range(len(frequencies)):
    group = {
        'lower': bin_edges[i],
        'upper': bin_edges[i + 1],
        'obs': float(frequencies[i]),
        'exp': float(expected_freq[i]),
    }

    if current is None:
        current = group
    else:
        current['upper'] = group['upper']
        current['obs'] += group['obs']
        current['exp'] += group['exp']

    # Close the group once expected frequency is large enough.
    if current['exp'] >= 5:
        merged_classes.append(current)
        current = None

# If a tail group remains with exp < 5, merge it into the previous group.
if current is not None:
    if merged_classes:
        merged_classes[-1]['upper'] = current['upper']
        merged_classes[-1]['obs'] += current['obs']
        merged_classes[-1]['exp'] += current['exp']
    else:
        merged_classes.append(current)

chi2_statistic = sum(
    ((grp['obs'] - grp['exp']) ** 2) / grp['exp']
    for grp in merged_classes
    if grp['exp'] > 0
)

# Degrees of freedom: k - 1 - p, with p=2 estimated parameters (mu and sigma).
k_merged = len(merged_classes)
df_chi2 = k_merged - 1 - 2

print(f"\n   Fréquences observées vs attendues (distribution normale):")
print(f"   {'Classe':<20} {'Observée':<12} {'Attendue':<12} {'| O-E |':<12}")
print(f"   {'-'*56}")

for grp in merged_classes:
    print(
        f"   [{grp['lower']:7.1f}, {grp['upper']:7.1f}]"
        f"   {grp['obs']:10.0f}   {grp['exp']:10.2f}   {abs(grp['obs']-grp['exp']):10.2f}"
    )

if df_chi2 > 0:
    chi2_critical = chi2.ppf(1 - alpha_test, df=df_chi2)
else:
    chi2_critical = float('nan')

print(f"\n   Statistique χ² (Pearson): {chi2_statistic:.6f}")
print(f"   Degrés de liberté: {df_chi2}")
print(f"   Valeur critique (α=5%): {chi2_critical:.6f}")

if df_chi2 <= 0:
    print("   Conclusion: Test non interprétable (ddl <= 0 après fusion des classes)")
elif chi2_statistic <= chi2_critical:
    print(f"   Condition: χ²₀ = {chi2_statistic:.6f} ≤ χ²_(1-α, ddl) = {chi2_critical:.6f}")
    print("   Conclusion: Les données SUIVENT une distribution normale")
else:
    print(f"   Condition: χ²₀ = {chi2_statistic:.6f} > χ²_(1-α, ddl) = {chi2_critical:.6f}")
    print("   Conclusion: Les données NE SUIVENT PAS une distribution normale")

# ============== PART IV: CONFIDENCE INTERVAL ==============
print("\n" + "=" * 70)
print("iv) INTERVALLE DE CONFIANCE POUR LA MOYENNE (95%)")
print("=" * 70)

confidence_level = 0.95
alpha_ci = 1 - confidence_level
z_critical = stats.norm.ppf(1 - alpha_ci/2)  # Z-score for normal distribution

# Using normal distribution
se_normal = std_dev / np.sqrt(n)
ci_lower_normal = mean - z_critical * se_normal
ci_upper_normal = mean + z_critical * se_normal

print(f"\nUtilisant la distribution NORMALE:")
print(f"   Moyenne: {mean:.2f} minutes")
print(f"   Écart-type: {std_dev:.2f} minutes")
print(f"   Taille de l'échantillon: {n}")
print(f"   Erreur standard (SE): σ/√n = {std_dev:.2f}/√{n} = {se_normal:.4f}")
print(f"   Z-critique (95%): {z_critical:.4f}")
print(f"   Intervalle de confiance: [{ci_lower_normal:.2f}, {ci_upper_normal:.2f}]")
print(f"   Interprétation: On est à 95% confiant que la vraie moyenne se situe")
print(f"                   entre {ci_lower_normal:.2f} et {ci_upper_normal:.2f} minutes")



# ============== PART V: HYPOTHESIS TEST ==============
print("\n" + "=" * 70)
print("v) TEST D'HYPOTHÈSE SUR LA MOYENNE (95% de confiance)")
print("=" * 70)

print(f"\nHypothèse du patron: Les joueurs jouent en MOYENNE au moins 300 minutes/semaine")
print(f"H₀: μ ≥ 300 (hypothèse nulle - hypothèse du patron)")
print(f"H₁: μ < 300 (hypothèse alternative)")
print(f"Type de test: Unilatéral à gauche")
print(f"\nNiveau de confiance: 95% → α = 5% = 0.05")

# Perform z-test manually using normal distribution
se = std_dev / np.sqrt(n)
z_stat = (mean - 300) / se  # z-statistic
z_critical_test = stats.norm.ppf(alpha_test)  # Critical z-value for left-tailed test (α = 0.05)

print(f"\nCalcul de la statistique z:")
print(f"   z₀ = (X̄ - μ₀) / SE")
print(f"   z₀ = ({mean:.2f} - 300) / {se:.4f}")
print(f"   z₀ = {z_stat:.6f}")

print(f"\nValeur critique:")
print(f"   z_α (pour α = {alpha_test}, test unilatéral à gauche) = {z_critical_test:.6f}")

print(f"\nComparaison:")
print(f"   z₀ = {z_stat:.6f}")
print(f"   z_α = {z_critical_test:.6f}")

print(f"\nDécision (α = {alpha_test}):")
if z_stat < z_critical_test:
    print(f"   z₀ ({z_stat:.6f}) < z_α ({z_critical_test:.6f})")
    print(f"   → REJETER H₀")
    print(f"   → Les données fournissent suffisamment de preuves que")
    print(f"     la moyenne est INFÉRIEURE à 300 minutes/semaine")
else:
    print(f"   z₀ ({z_stat:.6f}) ≥ z_α ({z_critical_test:.6f})")
    print(f"   → NE PAS REJETER H₀")
    print(f"   → Les données ne fournissent PAS suffisamment de preuves")
    print(f"     pour contredire le patron")

print(f"\nERREUR DE PREMIÈRE ESPÈCE (Type I Error):")
print(f"   α = {alpha_test} = 5%")
print(f"   Signification: Probabilité de rejeter H₀ alors qu'elle est vraie")
print(f"   En d'autres termes: 5% de chance de conclure à tort que")
print(f"   la moyenne est < 300 quand elle est vraiment ≥ 300")

# ============== PART VI: TYPE II ERROR ==============
print("\n" + "=" * 70)
print("vi) ERREUR DE DEUXIÈME ESPÈCE (Type II Error)")
print("=" * 70)

print(f"\nHypothèse: La moyenne échantillonnale est la vraie moyenne de population")
print(f"μ_population = {mean:.2f} minutes (estimée par la moyenne de l'échantillon)")

# Calculate the critical value under H₀ using normal distribution
z_critical_h0 = stats.norm.ppf(alpha_test)  # Critical z-score for α = 5%, left-tailed
se = std_dev / np.sqrt(n)
critical_value_mean = 300 + z_critical_h0 * se

print(f"\nSous H₀, la moyenne critique (seuil de rejet) est:")
print(f"   Moyenne critique = 300 + z_α × SE")
print(f"   Moyenne critique = 300 + ({z_critical_h0:.6f}) × {se:.4f}")
print(f"   Moyenne critique = {critical_value_mean:.2f} minutes")

# Calculate Type II error using normal distribution
# When true mean = sample_mean, we need P(not rejecting H₀ | H₁ is true)
z_beta = (critical_value_mean - mean) / se
beta = stats.norm.cdf(z_beta)

print(f"\nCalcul de β (erreur de deuxième espèce):")
print(f"   Vraie moyenne: μ = {mean:.2f}")
print(f"   H₀ rejetée si: X̄ < {critical_value_mean:.2f}")
print(f"   β = P(X̄ ≥ {critical_value_mean:.2f} | μ = {mean:.2f})")
print(f"   β = P(Z ≥ {z_beta:.6f})")
print(f"   β = {beta:.6f}")
print(f"   β ≈ {beta*100:.2f}%")

power = 1 - beta
print(f"\nPuissance du test (1 - β): {power:.6f} = {power*100:.2f}%")
print(f"Interprétation:")
print(f"   Si nous NE REJETONS PAS H₀, il y a une probabilité de {beta*100:.2f}%")
print(f"   que nous commettions une ERREUR DE DEUXIÈME ESPÈCE (faux négatif)")
print(f"   plutôt que d'affirmer correctement que μ < 300")

# ============== PART VII: VARIANCE TEST ==============
print("\n" + "=" * 70)
print("vii) TEST D'HYPOTHÈSE BILATÉRAL SUR LA VARIANCE")
print("=" * 70)

sigma_assumed = 50
sigma2_assumed = sigma_assumed ** 2

print(f"\nHypothèse: L'écart-type des temps de jeu est σ = {sigma_assumed} minutes")
print(f"H₀: σ² = {sigma2_assumed} (variance nulle)")
print(f"H₁: σ² ≠ {sigma2_assumed} (variance alternative - TEST BILATÉRAL)")
print(f"Niveau de signification: α = 5% = 0.05")

# Calculate chi-square statistic
chi2_stat = (n - 1) * variance / sigma2_assumed

print(f"\nCalcul de la statistique du khi-carré:")
print(f"   χ² = (n-1) × s² / σ₀²")
print(f"   χ² = ({n}-1) × {variance:.2f} / {sigma2_assumed}")
print(f"   χ² = {chi2_stat:.6f}")

# Critical values for two-tailed test
alpha_var = 0.05
chi2_lower = chi2.ppf(alpha_var/2, df=n-1)
chi2_upper = chi2.ppf(1 - alpha_var/2, df=n-1)

print(f"\nValeurs critiques (test bilatéral, α = {alpha_var}):")
print(f"   χ²_(α/2, df={n-1}) = {chi2_lower:.6f}")
print(f"   χ²_(1-α/2, df={n-1}) = {chi2_upper:.6f}")

print(f"\nDécision (α = {alpha_var}):")
if chi2_stat < chi2_lower or chi2_stat > chi2_upper:
    print(f"   χ² = {chi2_stat:.6f} est HORS de l'intervalle [{chi2_lower:.6f}, {chi2_upper:.6f}]")
    print(f"   → REJETER H₀")
    print(f"   → Les données fournissent suffisamment de preuves que")
    print(f"     la variance est DIFFÉRENTE de {sigma2_assumed}")
else:
    print(f"   χ² = {chi2_stat:.6f} est DANS l'intervalle [{chi2_lower:.6f}, {chi2_upper:.6f}]")
    print(f"   → NE PAS REJETER H₀")
    print(f"   → Les données ne fournissent PAS suffisamment de preuves")
    print(f"     pour contredire que σ² = {sigma2_assumed}")

print("\n" + "=" * 70)
