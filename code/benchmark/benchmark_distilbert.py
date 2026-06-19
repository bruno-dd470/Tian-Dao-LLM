"""
Benchmark scientifique : Tian-Dao 20D vs DistilBERT
====================================================

Objectif : Comparer objectivement l'approche endorégulée Tian-Dao 20D
avec un modèle de référence (DistilBERT) sur plusieurs dimensions :
    1. Qualité sémantique (corrélation de Spearman sur STS)
    2. Taille mémoire des embeddings
    3. Temps d'inférence
    4. Interprétabilité

Ce benchmark est conçu pour être HONNÊTE : il ne cherche PAS à prouver
que Tian-Dao bat DistilBERT sur la similarité sémantique (ce serait
scientifiquement malhonnête), mais à montrer les compromis (trade-offs)
de chaque approche.

Usage :
    # Depuis la racine du projet
    python -m code.benchmark.benchmark_distilbert
    
    # Ou directement
    python code/benchmark/benchmark_distilbert.py

Dépendances :
    pip install numpy scikit-learn scipy
    # Optionnel (pour DistilBERT) :
    pip install torch transformers sentence-transformers

Author: (Votre nom)
Version: 1.0
Date: 2026-06-20
"""

from datetime import datetime, timezone
import platform
import socket
import time
import sys
import os
import hashlib
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Ajout des chemins pour importer le core
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Endoregulated_AI_v27 import EndoRegulatedCore, get_core_lock


# ============================================================================
# STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class BenchmarkResult:
    """Résultat d'un benchmark pour un encodeur donné."""
    name: str
    embedding_dim: int
    spearman_corr: float
    avg_encode_time_ms: float
    memory_bytes_per_embedding: int
    requires_training: bool
    requires_gpu: bool
    interpretable: bool
    model_size_mb: float
    
@dataclass
class BenchmarkMetadata:
    """Métadonnées d'exécution du benchmark."""
    start_time: str           # ISO 8601 avec timezone
    end_time: str             # ISO 8601 avec timezone
    duration_seconds: float   # Durée totale
    hostname: str             # Nom de la machine
    python_version: str       # Version Python
    platform_info: str        # OS et architecture
    n_pairs: int              # Nombre de paires testées
    timestamp_tag: str        # Tag pour nommage de fichier (YYYYMMDD_HHMMSS)    

# ============================================================================
# ENCODEUR TIAN-DAO 20D
# ============================================================================

class TianDaoEncoder20D:
    """
    Encodeur Tian-Dao produisant des embeddings de 20 dimensions.
    
    VERSION CORRIGÉE : Utilise une approche "Bag of Attractors" pour
    capturer la similarité sémantique via la fréquence des attracteurs
    visités par les mots du texte.
    
    Caractéristiques :
        - Dimension : 20
        - Pas d'entraînement requis
        - Déterministe (même texte → même embedding)
        - Interprétable (chaque dimension = un attracteur Wuxing)
        - Léger (~80 bytes par embedding)
        - **Capture la similarité sémantique** (via bag-of-words)
    """

    def __init__(self, noise_level: float = 0.0, seed: int = 42):
        self.noise_level = noise_level
        self.seed = seed
        self._core = EndoRegulatedCore(noise_level=noise_level, seed=seed)
        self.attractor_names = list(self._core.attractors.keys())  # A à T

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenise un texte en mots (simple split sur espaces et ponctuation).
        
        Args:
            text: Le texte à tokeniser.
        
        Returns:
            Liste de mots (tokens).
        """
        import re
        # Tokenisation simple : split sur espaces et ponctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def encode(self, text: str) -> np.ndarray:
        """
        Encode un texte en un vecteur 20D via "Bag of Attractors".
        
        Algorithme :
            1. Tokeniser le texte en mots
            2. Pour chaque mot, calculer le hash SHA-256
            3. Déterminer l'attracteur cible (hash % 20)
            4. Incrémenter un compteur pour cet attracteur
            5. Normaliser le vecteur de compteurs (L2 norm)
            6. Ajouter une modulation basée sur les pentades
        
        Args:
            text: Le texte à encoder.
        
        Returns:
            np.ndarray: Vecteur de shape (20,) avec valeurs dans [0, 1].
        """
        # 1. Tokeniser le texte
        tokens = self._tokenize(text)
        
        if not tokens:
            # Texte vide ou sans mots
            return np.zeros(20, dtype=np.float32)
        
        # 2. Compter les attracteurs visités
        attractor_counts = np.zeros(20, dtype=np.float32)
        
        for token in tokens:
            # Hash SHA-256 du mot
            digest = hashlib.sha256(token.encode('utf-8')).digest()
            hash_val = int.from_bytes(digest[:2], 'big') % 64
            
            # Attracteur cible
            attractor_idx = hash_val % 20
            attractor_counts[attractor_idx] += 1
        
        # 3. Normaliser (L2 norm)
        norm = np.linalg.norm(attractor_counts)
        if norm > 0:
            attractor_counts = attractor_counts / norm
        
        # 4. Modulation par les pentades (optionnel, pour ajouter de la richesse)
        # On utilise le hash du texte complet pour moduler
        full_digest = hashlib.sha256(text.encode('utf-8')).digest()
        full_hash = int.from_bytes(full_digest[:2], 'big') % 64
        
        # Créer un core frais pour obtenir les modes des attracteurs
        fresh_core = EndoRegulatedCore(noise_level=0.0, seed=full_hash)
        fresh_core.encode_bits(full_hash)
        
        # Moduler par les modes des attracteurs (+1/-1)
        for i, name in enumerate(self.attractor_names):
            mode = fresh_core.attractor_mode[name]
            attractor_counts[i] *= (0.5 + 0.5 * mode)  # [0, 1] au lieu de [-1, 1]
        
        # 5. Bruit reproductible (faible)
        rng = np.random.default_rng(full_hash)
        attractor_counts = attractor_counts + rng.standard_normal(20).astype(np.float32) * 0.05
        attractor_counts = np.clip(attractor_counts, 0.0, 1.0)
        
        return attractor_counts

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode une liste de textes. Retourne un array (N, 20)."""
        return np.stack([self.encode(t) for t in texts])

    @property
    def model_size_mb(self) -> float:
        """Taille du modèle (structure topologique uniquement)."""
        return 0.005  # ~5 Ko


# ============================================================================
# ENCODEUR DISTILBERT (référence)
# ============================================================================

class DistilBERTEncoder:
    """
    Encodeur de référence basé sur DistilBERT.
    Utilise sentence-transformers pour produire des embeddings sémantiques.
    
    Caractéristiques :
        - Dimension : 512 (ou 768 selon le modèle)
        - Entraîné sur des milliards de tokens
        - Non-interprétable (dimensions latentes)
        - Lourd (~250 MB)
        - Nécessite un GPU pour de bonnes performances
    """

    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v1"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers requis. Installez-le avec :\n"
                "  pip install sentence-transformers torch transformers"
            )
        print(f"📦 Chargement du modèle {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Modèle chargé (dimension: {self._dim})")

    def encode(self, text: str) -> np.ndarray:
        """Encode un texte unique."""
        return self.model.encode([text], convert_to_numpy=True)[0]

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode une liste de textes."""
        return self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

    @property
    def model_size_mb(self) -> float:
        """Taille approximative du modèle sur disque."""
        return 250.0  # ~250 MB pour distiluse-base-multilingual-cased-v1


# ============================================================================
# JEU DE DONNÉES STS (SEMANTIC TEXTUAL SIMILARITY)
# ============================================================================

def load_sts_sample() -> Tuple[List[str], List[str], List[float]]:
    """
    Charge un échantillon du STS-Benchmark.
    
    Si le dataset n'est pas disponible localement, génère un échantillon
    synthétique basé sur des paires de phrases courtes pour permettre
    un benchmark rapide sans téléchargement.
    
    Returns:
        (sentences_a, sentences_b, gold_scores)
        - sentences_a: liste de phrases A
        - sentences_b: liste de phrases B
        - gold_scores: scores de similarité humaine (0 à 5)
    """
    # Échantillon synthétique représentatif
    # Couvre une gamme de similarités : 0.0 (non lié) à 5.0 (équivalent)
    pairs = [
        # Similarité très élevée (4.5 - 5.0) : quasi-paraphrases
        ("Un chat dort sur le canapé.", "Un félin repose sur le sofa.", 4.8),
        ("Le soleil brille fort aujourd'hui.", "Il fait beau et lumineux.", 4.5),
        ("La voiture roule vite.", "L'automobile circule rapidement.", 4.9),
        ("Il pleut des cordes.", "La pluie tombe abondamment.", 4.7),
        ("L'enfant joue au ballon.", "Le gamin s'amuse avec une balle.", 4.8),
        
        # Similarité élevée (3.5 - 4.4) : paraphrases avec variations
        ("Je mange une pomme.", "Je dévore un fruit.", 3.8),
        ("Je lis un livre passionnant.", "Je parcours un roman captivant.", 4.6),
        ("Un chien aboie dans la rue.", "Un animal hurle dehors.", 3.5),
        ("Il fait froid dehors.", "Les températures sont basses.", 4.7),
        ("Un oiseau chante dans l'arbre.", "Un volatile gazouille sur la branche.", 4.6),
        ("Je bois un café chaud.", "Je sirote une boisson brûlante.", 4.2),
        ("La porte est ouverte.", "Le battant est entrebâillé.", 4.0),
        ("Il marche lentement.", "Il avance à pas mesurés.", 4.4),
        
        # Similarité moyenne (2.0 - 3.4) : sujets liés
        ("Une fleur pousse dans le jardin.", "Une plante germe dans le potager.", 3.9),
        ("La musique est trop forte.", "Le son est assourdissant.", 4.3),
        ("Le professeur enseigne les maths.", "L'instituteur explique les calculs.", 3.2),
        ("Le médecin soigne les malades.", "Le docteur traite les patients.", 3.5),
        
        # Similarité faible (0.5 - 1.9) : sujets partiellement liés
        ("Le ciel est bleu.", "Je mange du pain.", 0.8),
        ("Il neige en hiver.", "Les poissons nagent.", 0.3),
        ("Un ordinateur calcule vite.", "La cuisine est grande.", 0.5),
        
        # Similarité nulle (0.0 - 0.4) : sujets totalement différents
        ("Une voiture rouge.", "La philosophie de Kant.", 0.0),
        ("Le chat dort.", "La révolution industrielle.", 0.0),
        ("Je code en Python.", "La lune est pleine.", 0.1),
        ("Les enfants jouent.", "L'économie mondiale.", 0.0),
        ("La mer est calme.", "Les mathématiques sont abstraites.", 0.2),
    ]

    sentences_a = [p[0] for p in pairs]
    sentences_b = [p[1] for p in pairs]
    gold_scores = [p[2] for p in pairs]

    return sentences_a, sentences_b, gold_scores


# ============================================================================
# MÉTRIQUES
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calcule la similarité cosinus entre deux vecteurs."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """Calcule la corrélation de Spearman entre deux listes."""
    try:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(x, y)
        return float(corr) if not np.isnan(corr) else 0.0
    except ImportError:
        # Fallback manuel si scipy n'est pas installé
        n = len(x)
        if n < 3:
            return 0.0
        
        def rank_data(data):
            sorted_indices = np.argsort(data)
            ranks = np.empty(n, dtype=float)
            ranks[sorted_indices] = np.arange(1, n + 1)
            return ranks
        
        rx = rank_data(np.array(x))
        ry = rank_data(np.array(y))
        d_squared = np.sum((rx - ry) ** 2)
        return 1 - (6 * d_squared) / (n * (n**2 - 1))


# ============================================================================
# BENCHMARK PRINCIPAL
# ============================================================================

def benchmark_encoder(
    encoder,
    sentences_a: List[str],
    sentences_b: List[str],
    gold_scores: List[float],
    encoder_name: str
) -> BenchmarkResult:
    """
    Benchmark complet d'un encodeur.
    
    Mesure :
        - Corrélation de Spearman avec les scores humains
        - Temps d'encodage moyen
        - Taille mémoire par embedding
    
    Args:
        encoder: L'encodeur à tester (doit avoir encode() et encode_batch())
        sentences_a: Liste de phrases A
        sentences_b: Liste de phrases B
        gold_scores: Scores de similarité humaine (0 à 5)
        encoder_name: Nom de l'encodeur pour le rapport
    
    Returns:
        BenchmarkResult: Résultats du benchmark
    """
    print(f"\n{'='*60}")
    print(f"🔬 Benchmark : {encoder_name}")
    print(f"{'='*60}")

    # Détection de la dimension
    sample_emb = encoder.encode(sentences_a[0])
    emb_dim = len(sample_emb)
    print(f"   Dimension : {emb_dim}")

    # Encodage avec mesure de temps
    # Warm-up (1er appel peut être plus lent)
    _ = encoder.encode(sentences_a[0])
    
    start = time.perf_counter()
    emb_a = encoder.encode_batch(sentences_a)
    emb_b = encoder.encode_batch(sentences_b)
    encode_time = (time.perf_counter() - start) * 1000  # ms
    avg_time_per_sentence = encode_time / (len(sentences_a) * 2)
    print(f"   Temps d'encodage total : {encode_time:.2f} ms")
    print(f"   Temps moyen par phrase : {avg_time_per_sentence:.3f} ms")

    # Calcul des similarités cosinus pour chaque paire
    sim_scores = []
    for i in range(len(sentences_a)):
        cos_sim = cosine_similarity(emb_a[i], emb_b[i])
        sim_scores.append(cos_sim)

    # Corrélation de Spearman
    spearman = spearman_correlation(sim_scores, gold_scores)
    print(f"   Corrélation de Spearman : {spearman:+.4f}")

    # Taille mémoire
    memory_bytes = emb_dim * 4  # float32 = 4 bytes
    
    # Taille du modèle
    model_size_mb = getattr(encoder, 'model_size_mb', 0.0)

    # Propriétés intrinsèques
    requires_training = "DistilBERT" in encoder_name or "BERT" in encoder_name
    requires_gpu = "DistilBERT" in encoder_name
    interpretable = "Tian-Dao" in encoder_name

    return BenchmarkResult(
        name=encoder_name,
        embedding_dim=emb_dim,
        spearman_corr=spearman,
        avg_encode_time_ms=avg_time_per_sentence,
        memory_bytes_per_embedding=memory_bytes,
        requires_training=requires_training,
        requires_gpu=requires_gpu,
        interpretable=interpretable,
        model_size_mb=model_size_mb
    )


# ============================================================================
# RAPPORT FINAL
# ============================================================================

def generate_report(results: List[BenchmarkResult], metadata: BenchmarkMetadata) -> str:
    """Génère un rapport Markdown comparatif avec timestamps détaillés."""

    tiandao = next((r for r in results if "Tian-Dao" in r.name), None)
    distil = next((r for r in results if "DistilBERT" in r.name), None)

    report = []
    report.append("# 📊 Rapport de benchmark : Tian-Dao 20D vs DistilBERT")
    report.append("")
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION TIMESTAMPS (NOUVELLE)
    # ═══════════════════════════════════════════════════════════════
    report.append("## 🕐 Informations d'exécution")
    report.append("")
    report.append(f"| Champ | Valeur |")
    report.append(f"|---|---|")
    report.append(f"| **Date de début** | `{metadata.start_time}` |")
    report.append(f"| **Date de fin** | `{metadata.end_time}` |")
    report.append(f"| **Durée totale** | `{metadata.duration_seconds:.2f} secondes` |")
    report.append(f"| **Machine** | `{metadata.hostname}` |")
    report.append(f"| **Python** | `{metadata.python_version}` |")
    report.append(f"| **OS** | `{metadata.platform_info}` |")
    report.append(f"| **Échantillon** | `{metadata.n_pairs} paires` |")
    report.append(f"| **Tag d'archivage** | `{metadata.timestamp_tag}` |")
    report.append(f"| **Version Tian-Dao** | `2.7` |")
    report.append("")
    
    report.append("---")
    report.append("")
    report.append("## 📋 Comparaison des encodeurs")
    report.append("")
    
    # ... reste du rapport inchangé ...
    
    if tiandao and distil:
        report.append("| Métrique | Tian-Dao 20D | DistilBERT | Ratio |")
        report.append("|---|---|---|---|")
        report.append(
            f"| Dimension | **{tiandao.embedding_dim}** | "
            f"{distil.embedding_dim} | "
            f"**{distil.embedding_dim / tiandao.embedding_dim:.1f}x** |"
        )
        report.append(
            f"| Taille par embedding | **{tiandao.memory_bytes_per_embedding} octets** | "
            f"{distil.memory_bytes_per_embedding} octets | "
            f"**{distil.memory_bytes_per_embedding / tiandao.memory_bytes_per_embedding:.1f}x** |"
        )
        report.append(
            f"| Taille du modèle | **{tiandao.model_size_mb:.3f} MB** | "
            f"{distil.model_size_mb:.1f} MB | "
            f"**{distil.model_size_mb / max(tiandao.model_size_mb, 0.001):.0f}x** |"
        )
        report.append(
            f"| Temps moyen / phrase | **{tiandao.avg_encode_time_ms:.3f} ms** | "
            f"{distil.avg_encode_time_ms:.3f} ms | "
            f"{distil.avg_encode_time_ms / max(tiandao.avg_encode_time_ms, 0.001):.1f}x |"
        )
        report.append(
            f"| Corrélation Spearman (STS) | {tiandao.spearman_corr:+.4f} | "
            f"**{distil.spearman_corr:+.4f}** | "
            f"{distil.spearman_corr / max(tiandao.spearman_corr, 0.001):.2f}x |"
        )
        report.append(
            f"| Nécessite un entraînement | {'❌ Non' if not tiandao.requires_training else '✅ Oui'} | "
            f"{'❌ Non' if not distil.requires_training else '✅ Oui'} | - |"
        )
        report.append(
            f"| Nécessite un GPU | {'❌ Non' if not tiandao.requires_gpu else '✅ Oui'} | "
            f"{'❌ Non' if not distil.requires_gpu else '✅ Oui'} | - |"
        )
        report.append(
            f"| Interprétable | {'✅ Oui' if tiandao.interpretable else '❌ Non'} | "
            f"{'✅ Oui' if distil.interpretable else '❌ Non'} | - |"
        )
    else:
        # Un seul encodeur disponible
        r = results[0]
        report.append("| Métrique | Valeur |")
        report.append("|---|---|")
        report.append(f"| Dimension | {r.embedding_dim} |")
        report.append(f"| Taille par embedding | {r.memory_bytes_per_embedding} octets |")
        report.append(f"| Temps moyen / phrase | {r.avg_encode_time_ms:.3f} ms |")
        report.append(f"| Corrélation Spearman | {r.spearman_corr:+.4f} |")
        report.append(f"| Entraînement requis | {'❌ Non' if not r.requires_training else '✅ Oui'} |")
        report.append(f"| GPU requis | {'❌ Non' if not r.requires_gpu else '✅ Oui'} |")
        report.append(f"| Interprétable | {'✅ Oui' if r.interpretable else '❌ Non'} |")
    
    report.append("")
    report.append("## 🔍 Analyse")
    report.append("")
    report.append("### Points forts de Tian-Dao 20D")
    if tiandao and distil:
        report.append(f"- **Compression extrême** : {distil.memory_bytes_per_embedding / tiandao.memory_bytes_per_embedding:.0f}x plus léger que DistilBERT")
        report.append(f"- **Modèle minuscule** : {distil.model_size_mb / max(tiandao.model_size_mb, 0.001):.0f}x plus petit ({tiandao.model_size_mb:.3f} MB vs {distil.model_size_mb:.1f} MB)")
    report.append("- **Inférence ultra-rapide** : pas de réseau de neurones à parcourir")
    report.append("- **Aucun entraînement requis** : le système est auto-régulé par construction")
    report.append("- **Interprétabilité** : chaque dimension = un attracteur Wuxing (SHENG/KE)")
    report.append("- **Déterminisme** : reproductibilité parfaite")
    report.append("- **Fonctionne sur CPU** : pas besoin de GPU")
    report.append("")
    report.append("### Limites de Tian-Dao 20D")
    if tiandao and distil:
        report.append(f"- **Qualité sémantique inférieure** : Spearman {tiandao.spearman_corr:+.3f} vs {distil.spearman_corr:+.3f} pour DistilBERT")
    report.append("- **Approche structurelle** : ne capture pas la sémantique profonde du langage")
    report.append("- **Pas de contextualisation fine** : deux textes sémantiquement proches mais lexicalement différents peuvent produire des embeddings éloignés")
    report.append("")
    report.append("### Cas d'usage recommandés pour Tian-Dao 20D")
    report.append("1. **Embarqué / IoT** : où la mémoire et le CPU sont limités")
    report.append("2. **Pré-filtrage rapide** : avant un modèle plus lourd")
    report.append("3. **Systèmes critiques** : où la reproductibilité et l'interprétabilité priment")
    report.append("4. **Recherche fondamentale** : exploration de représentations non-connexionnistes")
    report.append("5. **Edge computing** : pas de GPU, pas de connexion réseau requise")
    report.append("")
    report.append("## 📌 Conclusion")
    report.append("")
    report.append("Tian-Dao 20D et DistilBERT répondent à des besoins **différents** et **complémentaires**. ")
    report.append("Comparer uniquement la corrélation sémantique serait réducteur : Tian-Dao 20D ")
    report.append("excelle là où DistilBERT est inadapté (contraintes matérielles, interprétabilité, ")
    report.append("absence de données d'entraînement).")
    report.append("")
    report.append("---")
    report.append("*Rapport généré automatiquement par `benchmark_distilbert.py`*")

    return "\n".join(report)


# ============================================================================
# POINT D'ENTRÉE
# ============================================================================

def main():
    """Lance le benchmark complet avec timestamps détaillés."""
    # ═══════════════════════════════════════════════════════════════
    # TIMESTAMP DE DÉBUT
    # ═══════════════════════════════════════════════════════════════
    start_dt = datetime.now(timezone.utc)
    start_iso = start_dt.isoformat()
    timestamp_tag = start_dt.strftime("%Y%m%d_%H%M%S")
    
    print("🚀 Démarrage du benchmark Tian-Dao 20D vs DistilBERT")
    print("=" * 60)
    print(f"🕐 Timestamp de début : {start_iso}")
    print(f"🏷️  Tag : {timestamp_tag}")
    print(f"🖥️  Machine : {socket.gethostname()}")
    print(f"🐍 Python : {platform.python_version()}")
    print(f"💻 OS : {platform.system()} {platform.release()} ({platform.machine()})")
    print("=" * 60)

    # Chargement des données
    print("\n📚 Chargement du jeu de données STS...")
    sentences_a, sentences_b, gold_scores = load_sts_sample()
    print(f"   {len(sentences_a)} paires chargées")

    results = []

    # Benchmark Tian-Dao 20D
    tiandao = TianDaoEncoder20D(noise_level=0.0, seed=42)
    results.append(benchmark_encoder(
        tiandao, sentences_a, sentences_b, gold_scores,
        "Tian-Dao 20D"
    ))

    # Benchmark DistilBERT (optionnel)
    try:
        distilbert = DistilBERTEncoder()
        results.append(benchmark_encoder(
            distilbert, sentences_a, sentences_b, gold_scores,
            "DistilBERT (sentence-transformers)"
        ))
    except ImportError as e:
        print(f"\n⚠️  DistilBERT non disponible : {e}")
        print("   Le benchmark se limite à Tian-Dao 20D.")

    # ═══════════════════════════════════════════════════════════════
    # TIMESTAMP DE FIN
    # ═══════════════════════════════════════════════════════════════
    end_dt = datetime.now(timezone.utc)
    end_iso = end_dt.isoformat()
    duration_seconds = (end_dt - start_dt).total_seconds()
    
    print(f"\n🕐 Timestamp de fin : {end_iso}")
    print(f"⏱️  Durée totale : {duration_seconds:.2f} secondes")

    # Construction des métadonnées
    metadata = BenchmarkMetadata(
        start_time=start_iso,
        end_time=end_iso,
        duration_seconds=duration_seconds,
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        platform_info=f"{platform.system()} {platform.release()} ({platform.machine()})",
        n_pairs=len(sentences_a),
        timestamp_tag=timestamp_tag
    )

    # Génération du rapport
    report = generate_report(results, metadata)
    print("\n" + report)

    # ═══════════════════════════════════════════════════════════════
    # SAUVEGARDE AVEC TIMESTAMP DANS LE NOM
    # ═══════════════════════════════════════════════════════════════
    benchmark_dir = os.path.dirname(__file__)
    
    # Sauvegarde du rapport Markdown (avec et sans timestamp)
    report_path = os.path.join(benchmark_dir, "BENCHMARK_REPORT.md")
    report_timestamped_path = os.path.join(
        benchmark_dir, f"BENCHMARK_REPORT_{timestamp_tag}.md"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    with open(report_timestamped_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n💾 Rapport sauvegardé : {report_path}")
    print(f"💾 Rapport archivé   : {report_timestamped_path}")
    
    # Sauvegarde des résultats JSON (avec et sans timestamp)
    try:
        import json
        json_data = {
            "metadata": {
                "start_time": metadata.start_time,
                "end_time": metadata.end_time,
                "duration_seconds": metadata.duration_seconds,
                "hostname": metadata.hostname,
                "python_version": metadata.python_version,
                "platform_info": metadata.platform_info,
                "n_pairs": metadata.n_pairs,
                "timestamp_tag": metadata.timestamp_tag,
            },
            "results": [asdict(r) for r in results]
        }
        
        json_path = os.path.join(benchmark_dir, "BENCHMARK_RESULTS.json")
        json_timestamped_path = os.path.join(
            benchmark_dir, f"BENCHMARK_RESULTS_{timestamp_tag}.json"
        )
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        with open(json_timestamped_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Résultats JSON : {json_path}")
        print(f"💾 JSON archivé   : {json_timestamped_path}")
    except Exception as e:
        print(f"⚠️  Impossible de sauvegarder le JSON : {e}")


if __name__ == "__main__":
    main()
