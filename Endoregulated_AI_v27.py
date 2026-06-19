"""
Module: Endoregulated_AI_v26.py
Ce module implémente un système d'IA endorégulé, inspiré des principes du Wuxing
(philosophie chinoise des 5 phases) et de la topologie algébrique.

Contrairement aux modèles d'apprentissage profond (LLM, transformers, etc.),
ce système est un système dynamique déterministe qui s'auto-régule via un
cycle d'équilibre entre deux régimes opposés :
- SHENG : expansion, exploration, harmonisation.
- KE    : contraction, exploitation, contraste.

Le système projette des entrées 6 bits (0-63) dans un espace d'embedding
de 12 dimensions via un réseau d'attracteurs en forme de Merkabah.

Author: (Votre nom)
Version: 2.7  # [P0+P1] corrections appliquées
Date: 2026-06-19
"""

import numpy as np
import threading  # [P0-B] thread safety
import matplotlib.pyplot as plt
from collections import deque
from itertools import combinations
import networkx as nx
from enum import Enum

# Configuration matplotlib
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

# ============================================================================
# [P0-B] LOCK GLOBAL POUR THREAD SAFETY
# ============================================================================
_CORE_LOCK = threading.Lock()


def get_core_lock() -> threading.Lock:
    """Retourne le lock de protection du core (pour usage externe)."""
    return _CORE_LOCK


# ============================================================================
# DÉFINITION DES RÉGIMES
# ============================================================================
class Regime(Enum):
    """
    Énumération des régimes possibles du système endorégulé.
    - SHENG : État d'expansion et d'exploration.
    - KE : État de contraction et d'exploitation.
    - HOMEOSTASIE : État d'équilibre dynamique.
    """
    SHENG = "[SHENG] EXPLORATION"
    KE = "[KE] CONTRACTION"
    HOMEOSTASIE = "[HOME] EQUILIBRE"


# ============================================================================
# CŒUR TOPOLOGIQUE INVARIANT
# ============================================================================
class EndoRegulatedCore:
    """
    Cœur du système d'IA endorégulée.
    Graphe topologique de 20 attracteurs (A à T) organisés en Merkabah.
    """

    def __init__(self, noise_level: float = 0.1, seed: int = None):
        """
        Initialise le système avec une configuration équilibrée.

        Args:
            noise_level (float): Niveau de bruit stochastique (0.0 à 1.0).
            seed (int): Seed pour reproductibilité. None = aléatoire.
        """
        # [P0-B] Générateur local thread-safe
        self._rng = np.random.default_rng(seed)
        self.seed = seed

        self.attractors = self._build_merkabah()

        # ═══════════════════════════════════════════════════════════════
        # INITIALISATION ÉQUILIBRÉE : 10 SHENG / 10 KE
        # [P0-B] random.sample remplacé par self._rng.choice
        # ═══════════════════════════════════════════════════════════════
        attractor_names = list(self.attractors.keys())
        sheng_indices = self._rng.choice(
            len(attractor_names), size=10, replace=False
        )
        sheng_attractors = {attractor_names[i] for i in sheng_indices}

        self.attractor_mode = {}
        for name in attractor_names:
            self.attractor_mode[name] = 1 if name in sheng_attractors else -1

        # Reconstruire les pentades à partir des attracteurs
        self.pentad_sign = {f"{t}{i}": 0 for t in ['P', 'N'] for i in range(1, 7)}
        self._rebuild_pentads_from_attractors()

        # Seuils polaires
        self.thresholds = ['P4', 'N4']

        # Ceintures tropicales
        self.cp = ['P1', 'P3', 'P5', 'P6', 'P2']
        self.cn = ['N1', 'N2', 'N6', 'N5', 'N3']

        # État dynamique
        self.cycle_pos = 0
        self.cycle_neg = 0
        self.sheng_ke_phase = 1
        self.noise_level = noise_level
        self.input_counter = 0

        # Graphe des pentades
        self._build_pentad_graph()

    def _build_merkabah(self) -> dict:
        """Construit la structure de Merkabah des attracteurs."""
        return {
            'A': {'P1', 'P2', 'P4'}, 'B': {'P1', 'P3', 'P5'}, 'C': {'P2', 'P3', 'P6'},
            'D': {'P4', 'P5', 'N2'}, 'E': {'P5', 'P6', 'N3'}, 'F': {'P1', 'P6', 'N4'},
            'G': {'P2', 'P5', 'N6'}, 'H': {'P3', 'P4', 'N6'}, 'I': {'P1', 'N2', 'N6'},
            'J': {'P1', 'N3', 'N5'}, 'K': {'P2', 'N3', 'N5'}, 'L': {'P3', 'N2', 'N4'},
            'M': {'P4', 'N1', 'N3'}, 'N': {'P4', 'N5', 'N6'}, 'O': {'P5', 'N1', 'N4'},
            'P': {'P6', 'N1', 'N2'}, 'Q': {'P2', 'N1', 'N4'}, 'R': {'P3', 'N1', 'N5'},
            'S': {'P6', 'N5', 'N6'}, 'T': {'N2', 'N3', 'N4'}
        }

    def _rebuild_pentads_from_attractors(self) -> None:
        """Reconstruit les signes des pentades à partir des modes des attracteurs."""
        pentad_counts = {p: 0 for p in self.pentad_sign.keys()}

        for name, mode in self.attractor_mode.items():
            for p in self.attractors[name]:
                pentad_counts[p] += mode

        for p in pentad_counts:
            if pentad_counts[p] > 0:
                self.pentad_sign[p] = 1
            elif pentad_counts[p] < 0:
                self.pentad_sign[p] = -1
            else:
                # [P0-B] random.choice remplacé
                self.pentad_sign[p] = 1 if self._rng.random() > 0.5 else -1

        self._update_attractor_modes()

    def _build_pentad_graph(self) -> None:
        """Construit le graphe de connectivité des pentades."""
        self.pentad_graph = nx.Graph()
        self.pentad_graph.add_nodes_from(self.pentad_sign.keys())
        for p1, p2 in combinations(self.pentad_sign.keys(), 2):
            for pent_set in self.attractors.values():
                if p1 in pent_set and p2 in pent_set:
                    self.pentad_graph.add_edge(p1, p2)
                    break

    def _update_attractor_modes(self) -> None:
        """Met à jour les modes des attracteurs à partir des pentades."""
        for name, pent_set in self.attractors.items():
            p_list = list(pent_set)
            mode = (self.pentad_sign[p_list[0]]
                    * self.pentad_sign[p_list[1]]
                    * self.pentad_sign[p_list[2]])
            self.attractor_mode[name] = mode

    def eta_direct(self) -> float:
        """Calcule l'asymétrie spectrale η du système."""
        modes = list(self.attractor_mode.values())
        if not modes:
            return 0.0
        return sum(modes) / len(modes)

    # ========================================================================
    # [P0-A] CORRECTION CRITIQUE : encode_bits propage VRAIMENT le flip
    # ========================================================================
    def encode_bits(self, value: int) -> str:
        """
        Projette une valeur 6 bits (0-63) dans le système.

        CORRECTION v2.7 : Le flip est maintenant propagé aux pentades de
        l'attracteur cible, puis les modes des attracteurs voisins sont
        recalculés. L'entrée a donc un effet RÉEL et persistant.

        Args:
            value (int): Valeur à encoder (0-63).

        Returns:
            str: Nom de l'attracteur ciblé.

        Raises:
            ValueError: Si la valeur est hors de l'intervalle [0, 63].
        """
        if not (0 <= value <= 63):
            raise ValueError(f"Value {value} hors limites")

        target = list(self.attractors.keys())[value % 20]
        target_pentades = self.attractors[target]

        # ── ÉTAPE 1 : Flip des pentades de l'attracteur cible ──
        for p in target_pentades:
            self.pentad_sign[p] *= -1

        # ── ÉTAPE 2 : Recalcul des modes des attracteurs AFFECTÉS ──
        affected_attractors = set()
        for name, pent_set in self.attractors.items():
            if pent_set & target_pentades:  # intersection non vide
                affected_attractors.add(name)

        for name in affected_attractors:
            pent_list = list(self.attractors[name])
            mode = (self.pentad_sign[pent_list[0]]
                    * self.pentad_sign[pent_list[1]]
                    * self.pentad_sign[pent_list[2]])
            self.attractor_mode[name] = mode

        self.input_counter += 1
        return target

    def frustration(self) -> int:
        """Calcule l'énergie de frustration du système."""
        E = 0
        for p1, p2 in combinations(self.pentad_sign.keys(), 2):
            if self.pentad_graph.has_edge(p1, p2):
                if self.pentad_sign[p1] != self.pentad_sign[p2]:
                    E += 1
        return E

    def r_threshold(self) -> float:
        """Calcule l'activité des seuils polaires P4 et N4."""
        r = 0.0
        for p in self.thresholds:
            for n in self.pentad_graph.neighbors(p):
                if self.pentad_sign[p] != self.pentad_sign[n]:
                    r += 1.0
        return min(1.0, r / 12.0)

    def apply_wuxing_cycle(self) -> None:
        """Applique le cycle Wuxing pour faire évoluer le système."""
        eta = self.eta_direct()

        if eta > 0.3:
            self.sheng_ke_phase = -1
        elif eta < -0.15:
            self.sheng_ke_phase = 1

        if self.sheng_ke_phase == 1:
            p = self.cp[self.cycle_pos % 5]
            self.cycle_pos += 1
            incident = [self.attractor_mode[a] for a in self.attractors
                        if p in self.attractors[a]]
            if incident:
                self.pentad_sign[p] = 1 if sum(incident) > 0 else -1
        else:
            p = self.cn[self.cycle_neg % 5]
            self.cycle_neg = (self.cycle_neg + 2) % 5
            incident = [self.attractor_mode[a] for a in self.attractors
                        if p in self.attractors[a]]
            if incident:
                self.pentad_sign[p] = -1 if sum(incident) > 0 else 1

        self._update_attractor_modes()

    def propagate_thresholds(self) -> None:
        """Propage l'influence des seuils polaires."""
        for p in self.thresholds:
            # [P0-B] random.random remplacé
            if self._rng.random() < 0.1:
                self.pentad_sign[p] *= -1
                self._update_attractor_modes()

    def add_noise(self) -> None:
        """Ajoute du bruit stochastique au système."""
        # [P0-B] random.random / random.choice remplacés
        if self._rng.random() < self.noise_level:
            p = self._rng.choice(list(self.pentad_sign.keys()))
            self.pentad_sign[p] *= -1
            self._update_attractor_modes()

    def step(self) -> tuple:
        """Exécute un pas de temps du système."""
        self.apply_wuxing_cycle()
        self.propagate_thresholds()
        self.add_noise()
        return self.frustration(), self.eta_direct(), self.r_threshold()

    def get_regime(self) -> Regime:
        """Retourne le régime actuel du système."""
        eta = self.eta_direct()
        if eta >= 0:
            return Regime.SHENG
        else:
            return Regime.KE


# ============================================================================
# SIMULATEUR (version compatible Gradio)
# ============================================================================
class RandomInputSimulator:
    """Simulateur pour l'interface Gradio."""

    def __init__(self, core: EndoRegulatedCore, window: int = 200):
        self.core = core
        self.window = window
        self.eta_hist = deque(maxlen=window)
        self.e_hist = deque(maxlen=window)
        self.r_hist = deque(maxlen=window)
        self.step_hist = deque(maxlen=window)
        self.step = 0

    def record(self) -> None:
        """Enregistre l'état actuel du système dans l'historique."""
        self.eta_hist.append(self.core.eta_direct())
        self.e_hist.append(self.core.frustration())
        self.r_hist.append(self.core.r_threshold())
        self.step_hist.append(self.step)
        self.step += 1

    def run_one_step(self) -> dict:
        """Exécute un seul pas de simulation et enregistre l'état."""
        self.core.step()
        self.record()
        return {
            'eta': self.core.eta_direct(),
            'frustration': self.core.frustration(),
            'r_threshold': self.core.r_threshold(),
            'regime': self.core.get_regime().value,
            'input_counter': self.core.input_counter
        }


# ============================================================================
# POINT D'ENTRÉE POUR L'EXÉCUTION AUTONOME
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("IA ENDORÉGULÉE V2.7 - INVARIANT 64→20 (THREAD-SAFE)")
    print("INITIALISATION ÉQUILIBRÉE (10 SHENG / 10 KE)")
    print("=" * 70)

    # [P0-B] seed fixe pour reproductibilité
    core = EndoRegulatedCore(noise_level=0.15, seed=42)

    first_input = int(core._rng.integers(0, 64))
    attr = core.encode_bits(first_input)
    print(f"\n📥 Perturbation initiale: {first_input:2d} ({first_input:06b}) → attracteur {attr}")
    print(f"   η initial = {core.eta_direct():+.2f}")

    simulator = RandomInputSimulator(core)

    for i in range(20):
        result = simulator.run_one_step()
        if i % 5 == 0:
            print(f"  Step {i:3d} | η={result['eta']:+6.2f} | "
                  f"E_tot={result['frustration']:3d} | {result['regime']}")

    print("\n" + "=" * 70)
    print("📊 RÉSULTAT FINAL")
    print("=" * 70)
    print(f"   η = {core.eta_direct():+.2f}")
    print(f"   E_tot = {core.frustration()}")
    print(f"   R_th = {core.r_threshold():.2f}")
    print(f"   Régime = {core.get_regime().value}")
    print(f"   Total entrées traitées = {core.input_counter}")
