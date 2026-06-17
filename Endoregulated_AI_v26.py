"""
Module: Endoregulated_AI_v26.py

Ce module implémente un système d'IA endorégulé, inspiré des principes du Wuxing
(philosophie chinoise des 5 phases) et de la topologie algébrique.

Contrairement aux modèles d'apprentissage profond (LLM, transformers, etc.),
ce système est un **système dynamique déterministe** qui s'auto-régule via un
cycle d'équilibre entre deux régimes opposés :
    - SHENG : expansion, exploration, harmonisation.
    - KE    : contraction, exploitation, contraste.

Le système projette des entrées 6 bits (0-63) dans un espace d'embedding
de 12 dimensions via un réseau d'attracteurs en forme de Merkabah.

Ce code est conçu pour être exécuté en tant que bibliothèque pour l'interface
Gradio, ou en tant que script autonome pour des tests.

Author: (Votre nom)
Version: 2.6
Date: 2026-06-17
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from itertools import combinations
import networkx as nx
from enum import Enum

# Configuration matplotlib
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'


# ============================================================================
# DÉFINITION DES RÉGIMES
# ============================================================================

class Regime(Enum):
    """
    Énumération des régimes possibles du système endorégulé.

    - SHENG : État d'expansion et d'exploration. Le système cherche à
              harmoniser ses composants et à explorer de nouvelles configurations.
    - KE : État de contraction et d'exploitation. Le système se resserre
           sur des configurations stables et maximise le contraste.
    - HOMEOSTASIE : État d'équilibre dynamique où les deux forces s'équilibrent.
                    (Cet état est conceptuel, le système bascule toujours entre
                    SHENG et KE selon la valeur de η).
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

    Ce système est un graphe topologique composé de 20 attracteurs (A à T),
    organisés en une structure de Merkabah. Chaque attracteur est un triplet
    de pentades (P1-P6, N1-N6). L'état du système est défini par les signes
    (+1/-1) de ces 12 pentades.

    Le système évolue selon un cycle Wuxing, alternant entre les régimes
    SHENG et KE en fonction de l'asymétrie spectrale η.

    Attributes:
        attractors (dict): Dictionnaire des attracteurs et de leurs pentades.
        attractor_mode (dict): Mode (+1/-1) de chaque attracteur.
        pentad_sign (dict): Signe (+1/-1) de chaque pentade.
        thresholds (list): Seuils polaires ['P4', 'N4'].
        cp (list): Ceinture tropicale SHENG (pentagone).
        cn (list): Ceinture tropicale KE (pentagramme).
        cycle_pos (int): Position dans le cycle SHENG.
        cycle_neg (int): Position dans le cycle KE.
        sheng_ke_phase (int): Phase actuelle (1 = SHENG, -1 = KE).
        noise_level (float): Niveau de bruit stochastique.
        input_counter (int): Nombre total d'entrées traitées.
        pentad_graph (nx.Graph): Graphe de connectivité des pentades.
    """
    def __init__(self, noise_level: float = 0.1):
        """
        Initialise le système avec une configuration équilibrée.

        Le système est initialisé avec exactement 10 attracteurs en mode SHENG
        (+1) et 10 en mode KE (-1). Cette symétrie initiale est cruciale pour
        permettre une dynamique équilibrée et éviter un biais structurel.

        Args:
            noise_level (float): Niveau de bruit stochastique (0.0 à 1.0).
                                 Par défaut 0.1.
        """
        self.attractors = self._build_merkabah()

        # ═══════════════════════════════════════════════════════════════
        # INITIALISATION VRAIMENT ÉQUILIBRÉE : 10 SHENG / 10 KE
        # ═══════════════════════════════════════════════════════════════
        attractor_names = list(self.attractors.keys())
        sheng_attractors = set(random.sample(attractor_names, 10))

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
        """
        Construit la structure de Merkabah des attracteurs.

        La Merkabah est un graphe de 20 attracteurs (A à T), chacun étant
        un triplet de pentades. Cette structure est invariante et détermine
        les relations topologiques entre les pentades.

        Returns:
            dict: Dictionnaire des attracteurs et de leurs pentades.
        """
        return {
            'A': {'P1','P2','P4'}, 'B': {'P1','P3','P5'}, 'C': {'P2','P3','P6'},
            'D': {'P4','P5','N2'}, 'E': {'P5','P6','N3'}, 'F': {'P1','P6','N4'},
            'G': {'P2','P5','N6'}, 'H': {'P3','P4','N6'}, 'I': {'P1','N2','N6'},
            'J': {'P1','N3','N5'}, 'K': {'P2','N3','N5'}, 'L': {'P3','N2','N4'},
            'M': {'P4','N1','N3'}, 'N': {'P4','N5','N6'}, 'O': {'P5','N1','N4'},
            'P': {'P6','N1','N2'}, 'Q': {'P2','N1','N4'}, 'R': {'P3','N1','N5'},
            'S': {'P6','N5','N6'}, 'T': {'N2','N3','N4'}
        }

    def _rebuild_pentads_from_attractors(self) -> None:
        """
        Reconstruit les signes des pentades à partir des modes des attracteurs.

        Pour chaque pentade, on prend le signe majoritaire parmi les attracteurs
        qui la contiennent. En cas d'égalité, un tirage aléatoire est effectué.
        Cette méthode garantit que les pentades sont toujours cohérentes avec
        l'état des attracteurs.
        """
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
                self.pentad_sign[p] = random.choice([-1, 1])

        self._update_attractor_modes()

    def _build_pentad_graph(self) -> None:
        """
        Construit le graphe de connectivité des pentades.

        Deux pentades sont connectées si elles appartiennent au même attracteur.
        Ce graphe est utilisé pour calculer l'énergie de frustration du système.
        """
        self.pentad_graph = nx.Graph()
        self.pentad_graph.add_nodes_from(self.pentad_sign.keys())
        for p1, p2 in combinations(self.pentad_sign.keys(), 2):
            for pent_set in self.attractors.values():
                if p1 in pent_set and p2 in pent_set:
                    self.pentad_graph.add_edge(p1, p2)
                    break

    def _update_attractor_modes(self) -> None:
        """
        Met à jour les modes des attracteurs à partir des pentades.

        Le mode d'un attracteur est le produit des signes de ses trois pentades.
        Cette opération est l'inverse de _rebuild_pentads_from_attractors.
        """
        for name, pent_set in self.attractors.items():
            p_list = list(pent_set)
            mode = self.pentad_sign[p_list[0]] * self.pentad_sign[p_list[1]] * self.pentad_sign[p_list[2]]
            self.attractor_mode[name] = mode

    def eta_direct(self) -> float:
        """
        Calcule l'asymétrie spectrale η du système.

        Cette métrique est la moyenne des modes de tous les attracteurs.
        Elle représente l'état d'équilibre global du système dynamique :
            - η > 0  indique une dominance du régime SHENG (expansion/exploration).
            - η < 0  indique une dominance du régime KE (contraction/exploitation).
            - η ~ 0  indique un état d'homéostasie.

        Returns:
            float: L'asymétrie spectrale, comprise entre -1.0 et 1.0.
        """
        modes = list(self.attractor_mode.values())
        if not modes:
            return 0.0
        return sum(modes) / len(modes)

    def encode_bits(self, value: int) -> str:
        """
        Projette une valeur 6 bits (0-63) dans le système.

        Cette méthode est la porte d'entrée principale du système. Elle perturbe
        le système de manière minimale en inversant le signe de l'attracteur
        ciblé par la valeur d'entrée (via modulo 20).

        Cette perturbation minimale est inspirée du principe de "moindre action"
        et permet au système de s'adapter progressivement aux entrées.

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

        # Flip de l'attracteur cible
        self.attractor_mode[target] *= -1

        self._update_attractor_modes()
        self.input_counter += 1
        return target

    def frustration(self) -> int:
        """
        Calcule l'énergie de frustration du système.

        La frustration est le nombre d'arêtes du graphe des pentades dont
        les extrémités ont des signes opposés. C'est une mesure de la
        tension interne du système.

        Returns:
            int: Énergie de frustration.
        """
        E = 0
        for p1, p2 in combinations(self.pentad_sign.keys(), 2):
            if self.pentad_graph.has_edge(p1, p2):
                if self.pentad_sign[p1] != self.pentad_sign[p2]:
                    E += 1
        return E

    def r_threshold(self) -> float:
        """
        Calcule l'activité des seuils polaires P4 et N4.

        Cette métrique mesure la proportion de voisins des seuils polaires
        qui ont un signe opposé. Elle est utilisée pour détecter les
        pré-bifurcations dans le système.

        Returns:
            float: Activité des seuils, comprise entre 0.0 et 1.0.
        """
        r = 0.0
        for p in self.thresholds:
            for n in self.pentad_graph.neighbors(p):
                if self.pentad_sign[p] != self.pentad_sign[n]:
                    r += 1.0
        return min(1.0, r / 12.0)

    def apply_wuxing_cycle(self) -> None:
        """
        Applique le cycle Wuxing pour faire évoluer le système.

        Le cycle Wuxing alterne entre deux modes :
            - SHENG (pentagone) : harmonisation, exploration.
            - KE (pentagramme) : contraste, exploitation.

        La transition entre les deux modes est contrôlée par l'asymétrie η,
        avec des seuils asymétriques pour éviter les oscillations rapides.
        Cette hystérésis est inspirée des systèmes dynamiques biologiques
        et permet une stabilité globale.

        Les seuils sont les suivants :
            - η > 0.3  : transition vers KE.
            - η < -0.15 : transition vers SHENG.
        """
        eta = self.eta_direct()

        # Seuil asymétrique : plus difficile de rester en SHENG
        if eta > 0.3:
            self.sheng_ke_phase = -1  # Trop Sheng → on active Ke
        elif eta < -0.15:
            self.sheng_ke_phase = 1   # Trop Ke → on active Sheng

        if self.sheng_ke_phase == 1:
            # Mode Sheng : pentagone (harmonisation)
            p = self.cp[self.cycle_pos % 5]
            self.cycle_pos += 1
            incident = [self.attractor_mode[a] for a in self.attractors if p in self.attractors[a]]
            if incident:
                self.pentad_sign[p] = 1 if sum(incident) > 0 else -1
        else:
            # Mode Ke : pentagramme (contraste)
            p = self.cn[self.cycle_neg % 5]
            self.cycle_neg = (self.cycle_neg + 2) % 5
            incident = [self.attractor_mode[a] for a in self.attractors if p in self.attractors[a]]
            if incident:
                self.pentad_sign[p] = -1 if sum(incident) > 0 else 1

        self._update_attractor_modes()

    def propagate_thresholds(self) -> None:
        """
        Propage l'influence des seuils polaires.

        Avec une probabilité de 0.1, un seuil polaire (P4 ou N4) est inversé.
        Cette propagation permet aux seuils d'agir comme des "nœuds de contrôle"
        qui influencent l'ensemble du système.
        """
        for p in self.thresholds:
            if random.random() < 0.1:
                self.pentad_sign[p] *= -1
                self._update_attractor_modes()

    def add_noise(self) -> None:
        """
        Ajoute du bruit stochastique au système.

        Avec une probabilité égale au niveau de bruit, une pentade aléatoire
        est inversée. Ce bruit permet au système d'explorer de nouvelles
        configurations et d'éviter les minima locaux.
        """
        if random.random() < self.noise_level:
            p = random.choice(list(self.pentad_sign.keys()))
            self.pentad_sign[p] *= -1
            self._update_attractor_modes()

    def step(self) -> tuple:
        """
        Exécute un pas de temps du système.

        Un pas de temps inclut :
            1. Application du cycle Wuxing.
            2. Propagation des seuils.
            3. Ajout de bruit.

        Returns:
            tuple: (frustration, asymétrie η, activité des seuils R).
        """
        self.apply_wuxing_cycle()
        self.propagate_thresholds()
        self.add_noise()
        return self.frustration(), self.eta_direct(), self.r_threshold()

    def get_regime(self) -> str:
        """
        Retourne le régime actuel du système.

        Le régime est déterminé par le signe de l'asymétrie η :
            - η >= 0 : SHENG (exploration)
            - η < 0  : KE (contraction)

        Returns:
            str: Régime actuel sous forme de chaîne.
        """
        eta = self.eta_direct()
        if eta >= 0:
            return Regime.SHENG
        else:
            return Regime.KE


# ============================================================================
# SIMULATEUR (version compatible Gradio)
# ============================================================================

class RandomInputSimulator:
    """
    Simulateur pour l'interface Gradio.

    Cette classe encapsule le cœur endorégulé et fournit des fonctionnalités
    d'enregistrement et de visualisation des métriques. Elle est conçue pour
    être utilisée en mode pas-à-pas dans un environnement interactif.

    Attributes:
        core (EndoRegulatedCore): Le cœur du système.
        window (int): Taille de la fenêtre d'historique.
        eta_hist (deque): Historique de l'asymétrie η.
        e_hist (deque): Historique de la frustration E.
        r_hist (deque): Historique de l'activité des seuils R.
        step_hist (deque): Historique des pas de temps.
        step (int): Compteur de pas de temps.
    """
    def __init__(self, core: EndoRegulatedCore, window: int = 200):
        """
        Initialise le simulateur.

        Args:
            core (EndoRegulatedCore): Le cœur endorégulé.
            window (int): Taille de la fenêtre d'historique.
        """
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
        """
        Exécute un seul pas de simulation et enregistre l'état.

        Returns:
            dict: Dictionnaire contenant :
                - 'eta': Asymétrie spectrale.
                - 'frustration': Énergie de frustration.
                - 'r_threshold': Activité des seuils.
                - 'regime': Régime actuel.
                - 'input_counter': Nombre total d'entrées traitées.
        """
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
    """
    Point d'entrée pour l'exécution autonome du module.

    Cette section ne s'exécute que si le fichier est lancé directement,
    et non importé comme bibliothèque (par exemple dans Gradio).

    Elle lance une courte simulation de démonstration pour illustrer le
    fonctionnement du système.
    """
    random.seed(42)
    np.random.seed(42)

    print("=" * 70)
    print("IA ENDORÉGULÉE V2.6 - INVARIANT 64→20")
    print("INITIALISATION ÉQUILIBRÉE (10 SHENG / 10 KE)")
    print("=" * 70)

    core = EndoRegulatedCore(noise_level=0.15)

    first_input = random.randint(0, 63)
    attr = core.encode_bits(first_input)
    print(f"\n📥 Perturbation initiale: {first_input:2d} ({first_input:06b}) → attracteur {attr}")
    print(f"   η initial = {core.eta_direct():+.2f}")

    simulator = RandomInputSimulator(core)

    # Simulation non-bloquante : seulement quelques pas pour le test
    for i in range(20):
        result = simulator.run_one_step()
        if i % 5 == 0:
            print(f"  Step {i:3d} | η={result['eta']:+6.2f} | E_tot={result['frustration']:3d} | {result['regime']}")

    print("\n" + "=" * 70)
    print("📊 RÉSULTAT FINAL")
    print("=" * 70)
    print(f"   η = {core.eta_direct():+.2f}")
    print(f"   E_tot = {core.frustration()}")
    print(f"   R_th = {core.r_threshold():.2f}")
    print(f"   Régime = {core.get_regime().value}")
    print(f"   Total entrées traitées = {core.input_counter}")
