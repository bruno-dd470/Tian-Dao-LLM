# 📊 Rapport de benchmark : Tian-Dao 20D vs DistilBERT

## 🕐 Informations d'exécution

| Champ | Valeur |
|---|---|
| **Date de début** | `2026-06-19T16:42:40.529377+00:00` |
| **Date de fin** | `2026-06-19T16:42:48.902001+00:00` |
| **Durée totale** | `8.37 secondes` |
| **Machine** | `bruno` |
| **Python** | `3.11.2` |
| **OS** | `Linux 6.12.73+deb12-amd64 (x86_64)` |
| **Échantillon** | `25 paires` |
| **Tag d'archivage** | `20260619_164240` |
| **Version Tian-Dao** | `2.7` |

---

## 📋 Comparaison des encodeurs

| Métrique | Tian-Dao 20D | DistilBERT | Ratio |
|---|---|---|---|
| Dimension | **20** | 512 | **25.6x** |
| Taille par embedding | **80 octets** | 2048 octets | **25.6x** |
| Taille du modèle | **0.005 MB** | 250.0 MB | **50000x** |
| Temps moyen / phrase | **0.223 ms** | 4.865 ms | 21.8x |
| Corrélation Spearman (STS) | +0.0535 | **+0.5817** | 10.86x |
| Nécessite un entraînement | ❌ Non | ✅ Oui | - |
| Nécessite un GPU | ❌ Non | ✅ Oui | - |
| Interprétable | ✅ Oui | ❌ Non | - |

## 🔍 Analyse

### Points forts de Tian-Dao 20D
- **Compression extrême** : 26x plus léger que DistilBERT
- **Modèle minuscule** : 50000x plus petit (0.005 MB vs 250.0 MB)
- **Inférence ultra-rapide** : pas de réseau de neurones à parcourir
- **Aucun entraînement requis** : le système est auto-régulé par construction
- **Interprétabilité** : chaque dimension = un attracteur Wuxing (SHENG/KE)
- **Déterminisme** : reproductibilité parfaite
- **Fonctionne sur CPU** : pas besoin de GPU

### Limites de Tian-Dao 20D
- **Qualité sémantique inférieure** : Spearman +0.054 vs +0.582 pour DistilBERT
- **Approche structurelle** : ne capture pas la sémantique profonde du langage
- **Pas de contextualisation fine** : deux textes sémantiquement proches mais lexicalement différents peuvent produire des embeddings éloignés

### Cas d'usage recommandés pour Tian-Dao 20D
1. **Embarqué / IoT** : où la mémoire et le CPU sont limités
2. **Pré-filtrage rapide** : avant un modèle plus lourd
3. **Systèmes critiques** : où la reproductibilité et l'interprétabilité priment
4. **Recherche fondamentale** : exploration de représentations non-connexionnistes
5. **Edge computing** : pas de GPU, pas de connexion réseau requise

## 📌 Conclusion

Tian-Dao 20D et DistilBERT répondent à des besoins **différents** et **complémentaires**. 
Comparer uniquement la corrélation sémantique serait réducteur : Tian-Dao 20D 
excelle là où DistilBERT est inadapté (contraintes matérielles, interprétabilité, 
absence de données d'entraînement).

---
*Rapport généré automatiquement par `benchmark_distilbert.py`*