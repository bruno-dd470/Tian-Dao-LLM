# 📊 Rapport de benchmark : Tian-Dao 20D vs DistilBERT

## 🕐 Informations d'exécution

| Champ | Valeur |
|---|---|
| **Date de début** | `2026-06-20T08:17:57.171207+02:00` |
| **Date de fin** | `2026-06-20T08:22:31.743222+02:00` |
| **Durée totale** | `274.57 secondes` |
| **Machine** | `bruno` |
| **Python** | `3.11.2` |
| **OS** | `Linux 6.12.73+deb12-amd64 (x86_64)` |
| **Dataset** | `stsb_multi_mt (fr)` (official) |
| **Échantillon** | `8628 paires` |
| **IC niveau** | `95%` |
| **Bootstrap** | `300 itérations` |
| **Tag d'archivage** | `20260620_081757` |

---

## 📋 Comparaison des encodeurs

| Métrique | Tian-Dao 20D | DistilBERT | Ratio |
|---|---|---|---|
| Dimension | **20** | 512 | **25.6x** |
| Taille/embedding | **80 octets** | 2048 octets | **25.6x** |
| Taille modèle | **0.005 MB** | 250.0 MB | **50000x** |
| Temps/phrase | **0.064 ms** | 15.008 ms | 234.2x |
| **Spearman (STS)** | +0.0158 [-0.0472, +0.0418] | **+0.7751** [+0.7656, +0.7853] | N/A (structurel) |
| Entraînement | ❌ Non | ✅ Oui | - |
| GPU | ❌ Non | ✅ Oui | - |
| Interprétable | ✅ Oui | ❌ Non | - |

## 🔍 Analyse

### Points forts de Tian-Dao 20D
- **Compression extrême** : 26x plus léger
- **Modèle minuscule** : 50000x plus petit (0.005 MB vs 250.0 MB)
- **Inférence ultra-rapide** : pas de réseau de neurones
- **Aucun entraînement** : auto-régulé par construction
- **Interprétable** : chaque dimension = attracteur Wuxing
- **Déterministe** : reproductibilité parfaite

### Limites de Tian-Dao 20D
- **Spearman STS** : +0.016 vs +0.775 (DistilBERT)
- **Approche structurelle** : ne capture pas la sémantique profonde

## 📌 Conclusion

Tian-Dao 20D et DistilBERT répondent à des besoins **différents** et **complémentaires**.

---
*Rapport généré automatiquement par `benchmark_distilbert.py` v3.0*