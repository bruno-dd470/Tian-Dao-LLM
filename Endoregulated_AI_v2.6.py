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
# CŒUR TOPOLOGIQUE INVARIANT - INITIALISATION ÉQUILIBRÉE
# ============================================================================

class Regime(Enum):
    SHENG = "[SHENG] EXPLORATION"
    KE = "[KE] CONTRACTION"
    HOMEOSTASIE = "[HOME] EQUILIBRE"

class EndoRegulatedCore:
    def __init__(self, noise_level: float = 0.1):
        self.attractors = self._build_merkabah()
        
        # ═══════════════════════════════════════════════════════════════
        # INITIALISATION VRAIMENT ÉQUILIBRÉE : 10 SHENG / 10 KE
        # ═══════════════════════════════════════════════════════════════
        attractor_names = list(self.attractors.keys())
        # Choisir aléatoirement 10 attracteurs qui seront en mode SHENG (+1)
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
        
    def _build_merkabah(self):
        return {
            'A': {'P1','P2','P4'}, 'B': {'P1','P3','P5'}, 'C': {'P2','P3','P6'},
            'D': {'P4','P5','N2'}, 'E': {'P5','P6','N3'}, 'F': {'P1','P6','N4'},
            'G': {'P2','P5','N6'}, 'H': {'P3','P4','N6'}, 'I': {'P1','N2','N6'},
            'J': {'P1','N3','N5'}, 'K': {'P2','N3','N5'}, 'L': {'P3','N2','N4'},
            'M': {'P4','N1','N3'}, 'N': {'P4','N5','N6'}, 'O': {'P5','N1','N4'},
            'P': {'P6','N1','N2'}, 'Q': {'P2','N1','N4'}, 'R': {'P3','N1','N5'},
            'S': {'P6','N5','N6'}, 'T': {'N2','N3','N4'}
        }
    
    def _rebuild_pentads_from_attractors(self):
        """
        Reconstruit les signes des pentades à partir des modes des attracteurs.
        Pour chaque pentade, on prend le signe majoritaire parmi les attracteurs qui la contiennent.
        """
        # Initialisation des compteurs pour chaque pentade
        pentad_counts = {p: 0 for p in self.pentad_sign.keys()}
        
        # Pour chaque attracteur, on ajoute son mode aux pentades qui le composent
        for name, mode in self.attractor_mode.items():
            for p in self.attractors[name]:
                pentad_counts[p] += mode
        
        # Le signe de la pentade est le signe de la somme (majorité)
        for p in pentad_counts:
            if pentad_counts[p] > 0:
                self.pentad_sign[p] = 1
            elif pentad_counts[p] < 0:
                self.pentad_sign[p] = -1
            else:
                # Égalité : tirage aléatoire
                self.pentad_sign[p] = random.choice([-1, 1])
        
        self._update_attractor_modes()
    
    def _build_pentad_graph(self):
        self.pentad_graph = nx.Graph()
        self.pentad_graph.add_nodes_from(self.pentad_sign.keys())
        for p1, p2 in combinations(self.pentad_sign.keys(), 2):
            for pent_set in self.attractors.values():
                if p1 in pent_set and p2 in pent_set:
                    self.pentad_graph.add_edge(p1, p2)
                    break
    
    def _update_attractor_modes(self):
        """Met à jour les attracteurs = produit des 3 pentades"""
        for name, pent_set in self.attractors.items():
            p_list = list(pent_set)
            mode = self.pentad_sign[p_list[0]] * self.pentad_sign[p_list[1]] * self.pentad_sign[p_list[2]]
            self.attractor_mode[name] = mode
    
    def eta_direct(self) -> float:
        """Asymétrie globale : moyenne des modes des attracteurs"""
        modes = list(self.attractor_mode.values())
        if not modes:
            return 0.0
        return sum(modes) / len(modes)
    
    def encode_bits(self, value: int) -> str:
        """Perturbation minimale : projection 6 bits → attracteur"""
        if not (0 <= value <= 63):
            raise ValueError(f"Value {value} hors limites")
        
        target = list(self.attractors.keys())[value % 20]
        
        # UNIQUEMENT : flip de l'attracteur cible
        self.attractor_mode[target] *= -1
        
        self._update_attractor_modes()
        self.input_counter += 1
        return target
    
    def frustration(self) -> int:
        E = 0
        for p1, p2 in combinations(self.pentad_sign.keys(), 2):
            if self.pentad_graph.has_edge(p1, p2):
                if self.pentad_sign[p1] != self.pentad_sign[p2]:
                    E += 1
        return E
    
    def r_threshold(self) -> float:
        r = 0.0
        for p in self.thresholds:
            for n in self.pentad_graph.neighbors(p):
                if self.pentad_sign[p] != self.pentad_sign[n]:
                    r += 1.0
        return min(1.0, r / 12.0)
    
    def apply_wuxing_cycle(self):
        """Cycle Wuxing avec seuil asymétrique pour corriger le biais structurel"""
        eta = self.eta_direct()
        
        # Seuil asymétrique : plus difficile de rester en SHENG
        if eta > 0.3:           # Seuil plus haut pour quitter SHENG
            self.sheng_ke_phase = -1  # Trop Sheng → on active Ke
        elif eta < -0.15:        # Seuil plus bas pour quitter KE
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
    
    def propagate_thresholds(self):
        for p in self.thresholds:
            if random.random() < 0.1:
                self.pentad_sign[p] *= -1
                self._update_attractor_modes()
    
    def add_noise(self):
        if random.random() < self.noise_level:
            p = random.choice(list(self.pentad_sign.keys()))
            self.pentad_sign[p] *= -1
            self._update_attractor_modes()
    
    def step(self):
        self.apply_wuxing_cycle()
        self.propagate_thresholds()
        self.add_noise()
        return self.frustration(), self.eta_direct(), self.r_threshold()
    
    def get_regime(self) -> str:
        eta = self.eta_direct()
        if eta >= 0:
            return Regime.SHENG   # y compris quand eta est proche de 0
        else:
            return Regime.KE


# ============================================================================
# SIMULATEUR (version compatible, sans boucle interactive)
# ============================================================================

class RandomInputSimulator:
    """Version simplifiée pour l'interface Gradio"""
    def __init__(self, core: EndoRegulatedCore, window: int = 200):
        self.core = core
        self.window = window
        self.eta_hist = deque(maxlen=window)
        self.e_hist = deque(maxlen=window)
        self.r_hist = deque(maxlen=window)
        self.step_hist = deque(maxlen=window)
        self.step = 0
    
    def record(self):
        self.eta_hist.append(self.core.eta_direct())
        self.e_hist.append(self.core.frustration())
        self.r_hist.append(self.core.r_threshold())
        self.step_hist.append(self.step)
        self.step += 1
    
    def run_one_step(self):
        """Exécute un seul pas de simulation"""
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
# MAIN - MODIFIÉ POUR NE PAS BLOQUER
# ============================================================================

if __name__ == "__main__":
    # Ce code ne s'exécute QUE si le fichier est lancé directement
    # Il ne s'exécute PAS quand importé dans Gradio
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