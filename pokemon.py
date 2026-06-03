# ============================================
# PROGETTO POKEMON
# EDA + PCA + CLUSTERING
# ============================================

# Librerie
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ============================================
# 1. CARICAMENTO DATASET
# ============================================

df = pd.read_csv("Pokemon.csv")

print("Dimensioni dataset:")
print(df.shape)

print("\nPrime righe:")
print(df.head())

print("\nInformazioni dataset:")
print(df.info())

# ============================================
# 2. MISSING VALUES
# ============================================

print("\nValori mancanti:")
print(df.isnull().sum())

# ============================================
# 3. DUPLICATI
# ============================================

print("\nDuplicati:")
print(df.duplicated().sum())

# ============================================
# 4. STATISTICHE DESCRITTIVE
# ============================================

print("\nStatistiche descrittive:")
print(df.describe())

# ============================================
# 5. DISTRIBUZIONE VARIABILI NUMERICHE
# ============================================

stats = [
    'HP',
    'Attack',
    'Defense',
    'Sp. Atk',
    'Sp. Def',
    'Speed'
]

df[stats].hist(
    figsize=(12,8),
    bins=20
)

plt.suptitle("Distribuzione statistiche Pokemon")
plt.tight_layout()
plt.show()

# ============================================
# 6. CORRELAZIONE
# ============================================

plt.figure(figsize=(10,8))

corr = df[
    stats + ['Total']
].corr()

sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm'
)

plt.title("Matrice di correlazione")
plt.show()

# ============================================
# 7. ANALISI TIPI POKEMON
# ============================================

plt.figure(figsize=(10,5))

df["Type 1"].value_counts().plot(
    kind="bar"
)

plt.title("Distribuzione Type 1")
plt.show()

# ============================================
# 8. PREPARAZIONE DATI
# ============================================

# Eliminiamo:
# Name -> identificativo
# # -> numero pokedex
# Type1 e Type2 -> categoriche
# Legendary -> usata solo dopo

X = df[
    [
        'HP',
        'Attack',
        'Defense',
        'Sp. Atk',
        'Sp. Def',
        'Speed'
    ]
]

# ============================================
# 9. STANDARDIZZAZIONE
# ============================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================
# 10. PCA
# ============================================

pca = PCA()

X_pca = pca.fit_transform(X_scaled)

# ============================================
# 11. VARIANZA SPIEGATA
# ============================================

explained_variance = pca.explained_variance_ratio_

print("\nVarianza spiegata:")
print(explained_variance)

print("\nVarianza cumulata:")
print(np.cumsum(explained_variance))

# ============================================
# 12. SCREE PLOT
# ============================================

plt.figure(figsize=(8,5))

plt.plot(
    range(1,len(explained_variance)+1),
    explained_variance,
    marker='o'
)

plt.xlabel("Componenti")
plt.ylabel("Varianza spiegata")
plt.title("Scree Plot PCA")

plt.show()

# ============================================
# 13. PCA 2D
# ============================================

pca2 = PCA(n_components=2)

X_2d = pca2.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(
    X_2d[:,0],
    X_2d[:,1],
    alpha=0.6
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Proiezione PCA")

plt.show()

# ============================================
# 14. ELBOW METHOD
# ============================================

inertia = []

K = range(2,11)

for k in K:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    K,
    inertia,
    marker='o'
)

plt.xlabel("Numero cluster")
plt.ylabel("Inertia")

plt.title("Metodo Elbow")

plt.show()

# ============================================
# 15. SILHOUETTE SCORE
# ============================================

print("\nSilhouette Score")

for k in range(2,11):

    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = km.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(f"k={k} --> {score:.3f}")

# ============================================
# 16. KMEANS FINALE
# ============================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    X_scaled
)

# ============================================
# 17. VISUALIZZAZIONE CLUSTER
# ============================================

plt.figure(figsize=(8,6))

plt.scatter(
    X_2d[:,0],
    X_2d[:,1],
    c=df["Cluster"]
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("Cluster KMeans su PCA")

plt.show()

# ============================================
# 18. PROFILO CLUSTER
# ============================================

cluster_profile = df.groupby(
    "Cluster"
)[
    [
        'HP',
        'Attack',
        'Defense',
        'Sp. Atk',
        'Sp. Def',
        'Speed',
        'Total'
    ]
].mean()

print("\nProfilo cluster:")
print(cluster_profile)

# ============================================
# 19. LEGENDARY NEI CLUSTER
# ============================================

legendary_cluster = pd.crosstab(
    df["Cluster"],
    df["Legendary"]
)

print("\nLegendary per cluster:")
print(legendary_cluster)

# ============================================
# 20. CLUSTER GERARCHICO
# ============================================

agg = AgglomerativeClustering(
    n_clusters=4
)

labels_agg = agg.fit_predict(
    X_scaled
)

sil_agg = silhouette_score(
    X_scaled,
    labels_agg
)

print(
    "\nSilhouette Agglomerative:",
    round(sil_agg,3)
)

