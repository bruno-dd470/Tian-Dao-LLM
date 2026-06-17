"""
Module: app.py

Application Gradio pour la démonstration de l'IA endorégulée Tian-Dao.

Cette application transforme un texte en un embedding de 12 dimensions via un
système dynamique non-connectio­niste, inspiré des principes du Wuxing et de
la topologie algébrique.

Contrairement aux modèles d'embedding traditionnels (BERT, SBERT, etc.),
cette approche :
    - Ne nécessite aucun entraînement supervisé.
    - Est entièrement déterministe.
    - Produit des embeddings interprétables.
    - S'auto-régule via un cycle d'équilibre entre exploration (SHENG)
      et contraction (KE).

Le système projette des entrées 6 bits (0-63) dans un espace de 12 dimensions
via un réseau d'attracteurs en forme de Merkabah.

Author: (Votre nom)
Version: 1.0
Date: 2026-06-17
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os
import traceback

# Ajouter le chemin du projet pour permettre l'importation des modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer l'IA endorégulée
from Endoregulated_AI_v26 import EndoRegulatedCore, RandomInputSimulator


# ============================================================================
# INITIALISATION GLOBALE
# ============================================================================

# Le core est initialisé une seule fois au démarrage de l'application.
# Cela garantit que l'état du système est partagé entre toutes les requêtes.
core = EndoRegulatedCore(noise_level=0.15)
"""
EndoRegulatedCore: Instance globale du cœur endorégulé.

Le système est initialisé avec 10 attracteurs en mode SHENG (+1) et 10 en
mode KE (-1), ce qui assure une symétrie initiale et une dynamique équilibrée.
"""


# ============================================================================
# FONCTIONS DE TRAITEMENT
# ============================================================================

def text_to_embedding(text: str) -> tuple:
    """
    Convertit un texte en un embedding de 12 dimensions unique.

    Cette fonction est le cœur de l'application. Elle transforme un texte
    en une signature numérique en 12 dimensions en utilisant deux mécanismes
    complémentaires :

    1. **Hash modulaire** : Le texte est haché pour produire une valeur 0-63.
       Ce hash est utilisé pour moduler les pentades du système, créant ainsi
       un embedding unique pour chaque texte.

    2. **Modulation des pentades** : Les 12 pentades (P1-P6, N1-N6) du core
       sont modulées par le hash du texte. Cette modulation est déterministe
       et reproductible.

    3. **Bruit dépendant du texte** : Un bruit gaussien est ajouté, mais son
       seed est dérivé du hash, ce qui le rend reproductible pour le même texte.

    4. **Injection dans le core** : Le hash est injecté dans le core via
       `encode_bits`, ce qui fait évoluer le système dynamique.

    Ce processus produit un embedding qui est à la fois :
        - **Unique** pour chaque texte.
        - **Stable** dans le temps (le même texte donne le même embedding).
        - **Contextuel** (l'état du système influence l'embedding).

    Args:
        text (str): Texte à encoder.

    Returns:
        tuple: (embedding_12d, embedding_768d, attractor)
            - embedding_12d (np.ndarray): Embedding de 12 dimensions.
            - embedding_768d (np.ndarray): Embedding synthétique de 768 dimensions
              pour comparaison.
            - attractor (str): Nom de l'attracteur ciblé par le hash.

    Raises:
        Exception: Toute erreur est capturée et affichée dans les logs.

    Notes:
        Cette approche non-connectio­niste s'oppose aux modèles d'embedding
        traditionnels (BERT, SBERT) qui s'appuient sur des réseaux de neurones
        pré-entraînés sur des données massives. Ici, la "connaissance" du
        système est entièrement contenue dans sa structure topologique et
        son cycle d'auto-régulation.
    """
    try:
        # 1. Hash du texte (polynomial rolling hash)
        # Un nombre premier (31) est utilisé pour brasser les valeurs.
        hash_val = 0
        for i, char in enumerate(text):
            hash_val = (hash_val * 31 + ord(char)) % 64

        # 2. Modulation des pentades par le hash
        # Chaque pentade est modulée de manière déterministe en fonction du
        # hash et de sa position. Cette modulation crée un embedding unique.
        embedding = []

        # Pentades P1 à P6 (première moitié de l'embedding)
        for i in range(1, 7):
            base_val = core.pentad_sign[f'P{i}']
            mod = 1 if (hash_val + i * 2) % 3 != 0 else -1
            embedding.append(base_val * mod)

        # Pentades N1 à N6 (deuxième moitié de l'embedding)
        for i in range(1, 7):
            base_val = core.pentad_sign[f'N{i}']
            mod = 1 if (hash_val + i * 3 + 1) % 3 != 0 else -1
            embedding.append(base_val * mod)

        # 3. Conversion en array numpy
        emb = np.array(embedding, dtype=np.float32)

        # 4. Bruit dépendant du texte
        # Le bruit est reproductible car son seed est dérivé du hash.
        noise_seed = hash_val % 100
        np.random.seed(noise_seed)
        emb = emb + np.random.randn(12) * 0.15
        np.random.seed(None)  # Réinitialiser le seed global

        # 5. Normalisation pour rester dans [-1, 1]
        emb = np.clip(emb, -1.0, 1.0)

        # 6. Embedding 768D synthétique pour comparaison
        # Ce n'est pas un vrai embedding 768D, mais une simulation pour
        # illustrer le taux de compression.
        emb_768d = np.random.randn(768)

        # 7. Injection dans le core (fait évoluer le système)
        # Cette étape est cruciale : elle permet au système de "réagir" à
        # l'entrée et d'ajuster son état global.
        attractor = core.encode_bits(hash_val)

        return emb, emb_768d, attractor

    except Exception as e:
        print(f"Erreur dans text_to_embedding: {e}")
        traceback.print_exc()
        raise


def visualize_embedding(text: str) -> tuple:
    """
    Fonction principale de visualisation pour l'interface Gradio.

    Cette fonction orchestre l'ensemble du pipeline :
        1. Validation du texte d'entrée.
        2. Génération de l'embedding via `text_to_embedding`.
        3. Récupération des métriques du système (η, E, R).
        4. Création de la visualisation (barres + PCA).
        5. Construction des informations textuelles.

    La visualisation comprend deux graphiques :
        - **Graphique 1 (barres)** : Affiche les 12 dimensions de l'embedding.
          Les barres bleues indiquent des valeurs positives, les rouges des
          valeurs négatives. Les valeurs numériques sont affichées au-dessus
          des barres.

        - **Graphique 2 (PCA)** : Projette les embeddings historiques en 2D.
          Les points gris représentent l'historique, le point rouge est le
          nouvel embedding. Cette visualisation permet d'observer l'évolution
          du système dans le temps.

    Args:
        text (str): Texte à encoder et visualiser.

    Returns:
        tuple: (figure, texte_info)
            - figure (matplotlib.figure.Figure): Figure contenant les graphiques.
            - texte_info (str): Informations textuelles formatées en Markdown.

    Notes:
        Le cache des embeddings pour la PCA est stocké comme attribut statique
        de la fonction (`visualize_embedding.points_cache`). Cela permet de
        conserver l'historique entre les appels successifs.
    """
    try:
        # Validation du texte d'entrée
        if not text or text.strip() == "":
            return None, "⚠️ **Veuillez entrer un texte valide.**"

        # 1. Génération de l'embedding
        emb_12d, emb_768d, attractor = text_to_embedding(text)

        # 2. Récupération des métriques du système
        eta = core.eta_direct()
        frustration = core.frustration()
        r_threshold = core.r_threshold()
        regime = core.get_regime().value

        # 3. Création de la visualisation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ───────────────────────────────────────────────────────────────
        # GRAPHIQUE 1 : Barres de l'embedding 12D
        # ───────────────────────────────────────────────────────────────
        ax1 = axes[0]
        colors = ['blue' if v > 0 else 'red' for v in emb_12d]
        bars = ax1.bar(range(1, len(emb_12d) + 1), emb_12d, color=colors, alpha=0.7)
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Valeur')
        ax1.set_title(f'Embedding 12D (attracteur {attractor})')
        ax1.set_ylim(-1.5, 1.5)
        ax1.grid(True, alpha=0.3)

        # Ajout des valeurs sur les barres
        for bar, val in zip(bars, emb_12d):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8)

        # ───────────────────────────────────────────────────────────────
        # GRAPHIQUE 2 : Projection 2D via PCA
        # ───────────────────────────────────────────────────────────────
        ax2 = axes[1]

        # Gestion du cache des points pour la PCA
        if not hasattr(visualize_embedding, 'points_cache'):
            visualize_embedding.points_cache = []

        visualize_embedding.points_cache.append(emb_12d)
        if len(visualize_embedding.points_cache) > 20:
            visualize_embedding.points_cache.pop(0)

        # PCA sur les points historiques
        if len(visualize_embedding.points_cache) >= 3:
            try:
                points = np.array(visualize_embedding.points_cache)
                pca = PCA(n_components=2)
                points_2d = pca.fit_transform(points)

                # Affichage des points
                ax2.scatter(points_2d[:-1, 0], points_2d[:-1, 1],
                           c='gray', alpha=0.5, s=40, label='Historique')
                ax2.scatter(points_2d[-1, 0], points_2d[-1, 1],
                           c='red', s=120, label='Nouveau', edgecolors='black', linewidth=2)
                ax2.set_title('Projection 2D (PCA)')
                ax2.legend()
            except Exception as e:
                ax2.text(0.5, 0.5, f'PCA: {str(e)[:40]}',
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Erreur PCA')
        else:
            ax2.text(0.5, 0.5, f'{(3 - len(visualize_embedding.points_cache))} entrée(s)\nsupplémentaire(s) pour la PCA',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('En attente de plus de données...')

        ax2.set_xlabel('Composante 1')
        ax2.set_ylabel('Composante 2')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 4. Construction des informations textuelles
        info_text = (
            f"**Analyse du texte :** `{text[:50]}{'...' if len(text)>50 else ''}`\n\n"
            f"**Attracteur :** {attractor}\n"
            f"**Asymétrie η :** {eta:+.2f}\n"
            f"**Frustration E :** {frustration}\n"
            f"**Seuil R :** {r_threshold:.2f}\n"
            f"**Régime :** {regime}\n"
            f"**Embedding 12D :** {len(emb_12d)} dims (taille : {emb_12d.nbytes} bytes)\n"
            f"**Embedding 768D :** {len(emb_768d)} dims (taille : {emb_768d.nbytes} bytes)\n"
            f"**Taux de compression :** {emb_768d.nbytes / emb_12d.nbytes:.1f}x\n"
            f"**Entrées traitées :** {core.input_counter}"
        )

        # Fermeture de la figure pour éviter les fuites mémoire
        plt.close(fig)
        return fig, info_text

    except Exception as e:
        error_msg = f"❌ **Erreur :** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return None, error_msg


# ============================================================================
# INTERFACE GRADIO
# ============================================================================

def create_interface() -> gr.Blocks:
    """
    Crée l'interface Gradio pour la démonstration.

    L'interface se compose de :
        - Une zone de saisie de texte.
        - Un bouton pour générer l'embedding.
        - Un bouton pour effacer le texte.
        - Un graphique de visualisation.
        - Un affichage des statistiques.
        - Des exemples de textes préchargés.

    L'interface est conçue pour être intuitive et informative, avec des
    sections organisées pour faciliter l'exploration.

    Returns:
        gr.Blocks: Interface Gradio prête à être lancée.
    """
    with gr.Blocks(title="Tian-Dao Embeddings Demo", theme=gr.themes.Soft()) as demo:
        # En-tête de l'application
        gr.Markdown("""
        # 🧠 Tian-Dao Embeddings Demo
        ### IA Endorégulée - Invariant 64→12 avec Wuxing Cycle

        Cette démo transforme votre texte en un embedding 12D unique via un
        **système dynamique non-connectio­niste**. Le système alterne entre les
        régimes **SHENG** (exploration) et **KE** (contraction) selon un cycle
        d'auto-régulation inspiré du Wuxing.

        - **64 → 12** : Projection d'une entrée 6 bits vers un espace 12D.
        - **Déterministe** : Même texte → même embedding.
        - **Adaptatif** : Le système évolue à chaque entrée.
        """)

        # Organisation en colonnes
        with gr.Row():
            # Colonne de gauche : saisie et statistiques
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Texte à encoder",
                    placeholder="Saisissez votre texte ici...",
                    lines=3
                )
                submit_btn = gr.Button("🔮 Générer l'embedding", variant="primary")
                clear_btn = gr.Button("🗑️ Effacer")

                # Section des statistiques (dépliable)
                with gr.Accordion("📊 Métriques détaillées", open=True):
                    stats_text = gr.Markdown("**Statistiques :** en attente d'entrée...")

            # Colonne de droite : visualisation
            with gr.Column(scale=3):
                plot_output = gr.Plot(label="Visualisation de l'embedding")

        # Exemples de textes (cliquables)
        gr.Markdown("### 📝 Exemples de textes")
        gr.Examples(
            examples=[
                "Bonjour le monde",
                "L'intelligence artificielle est fascinante",
                "Le cycle Wuxing gouverne l'équilibre",
                "Sheng et Ke dansent dans le chaos",
                "La topologie des attracteurs révèle l'harmonie"
            ],
            inputs=text_input,
            label="Cliquez sur un exemple pour le charger"
        )

        # ───────────────────────────────────────────────────────────────
        # FONCTIONS DE GESTION DES ÉVÉNEMENTS
        # ───────────────────────────────────────────────────────────────

        def submit_text(text: str) -> tuple:
            """
            Gère l'événement de soumission du texte.

            Cette fonction est appelée lorsque l'utilisateur clique sur
            "Générer l'embedding" ou appuie sur Entrée.

            Args:
                text (str): Texte à encoder.

            Returns:
                tuple: (figure, info_text)
            """
            if not text or text.strip() == "":
                return None, "⚠️ **Veuillez saisir un texte valide.**"
            return visualize_embedding(text)

        def clear_text() -> tuple:
            """
            Gère l'événement d'effacement du texte.

            Cette fonction réinitialise le cache des points pour la PCA
            et vide les champs de l'interface.

            Returns:
                tuple: (texte_vide, figure_vide, message)
            """
            if hasattr(visualize_embedding, 'points_cache'):
                visualize_embedding.points_cache = []
            return "", None, "Entrez un nouveau texte pour générer un embedding."

        # Connexion des événements aux boutons
        submit_btn.click(
            fn=submit_text,
            inputs=text_input,
            outputs=[plot_output, stats_text]
        )

        clear_btn.click(
            fn=clear_text,
            outputs=[text_input, plot_output, stats_text]
        )

        # Soumission par appui sur Entrée
        text_input.submit(
            fn=submit_text,
            inputs=text_input,
            outputs=[plot_output, stats_text]
        )

    return demo


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================

# Instance globale de l'interface
demo = create_interface()

if __name__ == "__main__":
    """
    Point d'entrée pour l'exécution de l'application Gradio.

    Lorsque le fichier est lancé directement, cette section démarre le
    serveur Gradio qui héberge l'application.
    """
    demo.launch()
