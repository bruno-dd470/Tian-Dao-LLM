"""
Module: app.py
Application Gradio pour la démonstration de l'IA endorégulée Tian-Dao.
Version: 1.2  # [P0+P1] corrections appliquées
Date: 2026-06-20
"""

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')  # [P3] Backend non-interactive pour tests/serveur
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hashlib  # [P1-3] hash robuste
import traceback

# [P1-2] Import direct depuis la racine
from Endoregulated_AI_v27 import EndoRegulatedCore, RandomInputSimulator, get_core_lock


# ============================================================================
# [P0-C] INITIALISATION GLOBALE THREAD-SAFE
# ============================================================================
core = EndoRegulatedCore(noise_level=0.15, seed=42)
core_lock = get_core_lock()


# ============================================================================
# FONCTIONS DE TRAITEMENT
# ============================================================================
def text_to_embedding(text: str) -> tuple:
    """
    Convertit un texte en un embedding de 20 dimensions unique.
    Chaque dimension correspond à un attracteur (classe d'équivalence
    topologique du schème 64→20 de Cl(6,0)).
    
    Returns:
        tuple: (embedding_20d, embedding_768d, attractor)
    """
    try:
        # [P1-3] Hash SHA-256 tronqué sur 6 bits
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        hash_val = int.from_bytes(digest[:2], 'big') % 64
        
        # 20 triplets de pentades (Table 1 du PDF complexity.pdf)
        # Chaque triplet identifie un attracteur de la Merkabah
        ATTRACTOR_TRIPLETS = [
            # 3P (3 classes) - pôles géométriquement isolés
            ['P1', 'P2', 'P4'],  # A: Methionine
            ['P1', 'P3', 'P5'],  # B: Tryptophan
            ['P2', 'P3', 'P6'],  # C: Phenylalanine
            # 2P+1N (5 classes) - faces/arrêtes primaires
            ['P4', 'P5', 'N2'],  # D: Isoleucine
            ['P5', 'P6', 'N3'],  # E: Valine
            ['P1', 'P6', 'N4'],  # F: Proline
            ['P2', 'P5', 'N6'],  # G: Threonine
            ['P3', 'P4', 'N6'],  # H: Alanine
            # 1P+2N (11 classes) - sommets/diagonales/intersections
            ['P1', 'N2', 'N6'],  # I: Serine
            ['P1', 'N3', 'N5'],  # J: Leucine
            ['P2', 'N3', 'N5'],  # K: Arginine
            ['P3', 'N2', 'N4'],  # L: Glycine
            ['P4', 'N1', 'N3'],  # M: Tyrosine
            ['P4', 'N5', 'N6'],  # N: Histidine
            ['P5', 'N1', 'N4'],  # O: Glutamine
            ['P6', 'N1', 'N2'],  # P: Asparagine
            ['P2', 'N1', 'N4'],  # Q: Lysine
            ['P3', 'N1', 'N5'],  # R: Aspartic acid
            ['P6', 'N5', 'N6'],  # S: Glutamic acid
            # 3N (1 classe) - coeur interne (seuil fonctionnel)
            ['N2', 'N3', 'N4'],  # T: Cysteine + STOP
        ]
        
        # Calcul de l'embedding 20D
        # Chaque dimension = signature de polarité d'un attracteur
        embedding = []
        for triplet in ATTRACTOR_TRIPLETS:
            # Compter les pentades positives et négatives
            n_positive = sum(1 for p in triplet if p.startswith('P'))
            n_negative = sum(1 for p in triplet if p.startswith('N'))
            
            # Signature de polarité normalisée [-1, +1]
            # 3P → +1.0, 2P+1N → +0.333, 1P+2N → -0.333, 3N → -1.0
            polarity_score = (n_positive - n_negative) / 3.0
            
            # Modulation par le hash pour variabilité spécifique au texte
            mod = 1.0 if (hash_val + len(embedding)) % 5 != 0 else -1.0
            embedding.append(polarity_score * mod)
        
        # [P3] dtype=np.float32 explicite
        emb = np.array(embedding, dtype=np.float32)
        
        # [P0-C] Bruit déterministe — THREAD-SAFE
        rng = np.random.default_rng(hash_val)
        emb = emb + rng.standard_normal(20).astype(np.float32) * 0.15
        
        # Normalisation
        emb = np.clip(emb, -1.0, 1.0)
        
        # [P0-C] Embedding 768D DÉTERMINISTE
        rng_768 = np.random.default_rng(hash_val + 1000)
        emb_768d = rng_768.standard_normal(768).astype(np.float32)
        
        # [P0-C] Injection dans le core — PROTÉGÉE PAR LOCK
        with core_lock:
            attractor = core.encode_bits(hash_val)
        
        return emb, emb_768d, attractor
    
    except Exception as e:
        print(f"Erreur dans text_to_embedding: {e}")
        traceback.print_exc()
        raise


def visualize_embedding(text: str, points_cache: list) -> tuple:
    """
    Fonction principale de visualisation pour l'interface Gradio.
    Args:
        text (str): Texte à encoder.
        points_cache (list): Cache des embeddings pour la PCA (par session).
    Returns:
        tuple: (figure, texte_info, points_cache_mis_à_jour)
    """
    fig = None
    try:
        # [P1-5] Validation robuste
        if not text or not text.strip():
            return None, "⚠️ **Veuillez entrer un texte valide.**", points_cache

        # 1. Génération de l'embedding
        emb_20d, emb_768d, attractor = text_to_embedding(text)

        # 2. [P0-C] Récupération des métriques — PROTÉGÉE PAR LOCK
        with core_lock:
            eta = core.eta_direct()
            frustration = core.frustration()
            r_threshold = core.r_threshold()
            regime = core.get_regime().value
            input_counter = core.input_counter

        # 3. Création de la visualisation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ── GRAPHIQUE 1 : Barres de l'embedding 20D ──
        ax1 = axes[0]
        colors = ['blue' if v > 0 else 'red' for v in emb_20d]
        bars = ax1.bar(range(1, len(emb_20d) + 1), emb_20d, color=colors, alpha=0.7)
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Valeur')
        ax1.set_title(f'Embedding 20D (attracteur {attractor})')
        ax1.set_ylim(-1.5, 1.5)
        ax1.grid(True, alpha=0.3)

        for bar, val in zip(bars, emb_20d):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.1f}',
                     ha='center',
                     va='bottom' if height > 0 else 'top',
                     fontsize=8)

        # ── GRAPHIQUE 2 : Projection 2D via PCA ──
        ax2 = axes[1]

        if points_cache is None:
            points_cache = []

        points_cache.append(emb_20d.copy())  # [P3] copie pour éviter références partagées
        if len(points_cache) > 20:
            points_cache.pop(0)

        # [P1-1] Seuil minimum de 5 points pour PCA stable
        if len(points_cache) >= 5:
            try:
                points = np.array(points_cache)
                if np.linalg.matrix_rank(points) < 2:
                    ax2.text(0.5, 0.5, 'Données colinéaires\nen attente de variance...',
                             ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('PCA en attente')
                else:
                    pca = PCA(n_components=2)
                    points_2d = pca.fit_transform(points)

                    ax2.scatter(points_2d[:-1, 0], points_2d[:-1, 1],
                                c='gray', alpha=0.5, s=40, label='Historique')
                    ax2.scatter(points_2d[-1, 0], points_2d[-1, 1],
                                c='red', s=120, label='Nouveau',
                                edgecolors='black', linewidth=2)
                    ax2.set_title('Projection 2D (PCA)')
                    ax2.legend()
            except Exception as e:
                ax2.text(0.5, 0.5, f'PCA: {str(e)[:40]}',
                         ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Erreur PCA')
        else:
            ax2.text(0.5, 0.5,
                     f'{(5 - len(points_cache))} entrée(s)\n'
                     f'supplémentaire(s) pour la PCA',
                     ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12)
            ax2.set_title('En attente de plus de données...')

        ax2.set_xlabel('Composante 1')
        ax2.set_ylabel('Composante 2')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 4. Construction des informations textuelles
        info_text = (
            f"**Analyse du texte :** `{text[:50]}{'...' if len(text) > 50 else ''}`\n\n"
            f"**Attracteur :** {attractor}\n"
            f"**Asymétrie η :** {eta:+.2f}\n"
            f"**Frustration E :** {frustration}\n"
            f"**Seuil R :** {r_threshold:.2f}\n"
            f"**Régime :** {regime}\n"
            f"**Embedding 20D :** {len(emb_20d)} dims "
            f"(taille : {emb_20d.nbytes} bytes)\n"
            f"**Embedding 768D :** {len(emb_768d)} dims "
            f"(taille : {emb_768d.nbytes} bytes)\n"
            f"**Taux de compression :** {emb_768d.nbytes / emb_20d.nbytes:.1f}x\n"
            f"**Entrées traitées :** {input_counter}"
        )

        return fig, info_text, points_cache

    except Exception as e:
        error_msg = (f"❌ **Erreur :** {str(e)}\n\n"
                     f"```\n{traceback.format_exc()}\n```")
        return None, error_msg, points_cache if points_cache else []

    finally:
        # [P1-6] Fermeture systématique
        if fig is not None:
            plt.close(fig)


# ============================================================================
# INTERFACE GRADIO
# ============================================================================
def create_interface() -> gr.Blocks:
    """Crée l'interface Gradio pour la démonstration."""
    # [P3] Gradio 6.0 : theme passé à launch()
    with gr.Blocks(title="Tian-Dao Embeddings Demo") as demo:

        gr.Markdown("""
        # 🧠 Tian-Dao Embeddings Demo
        ### IA Endorégulée - Invariant 64→20 avec Wuxing Cycle

        Cette démo transforme votre texte en un embedding 20D unique via un
        **système dynamique non-connexionniste**. Le système alterne entre les
        régimes **SHENG** (exploration) et **KE** (contraction) selon un cycle
        d'auto-régulation inspiré du Wuxing.
        """)

        # [P0-D] État par session utilisateur
        points_state = gr.State(value=[])

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Texte à encoder",
                    placeholder="Saisissez votre texte ici...",
                    lines=3
                )
                submit_btn = gr.Button("🔮 Générer l'embedding",
                                       variant="primary")
                clear_btn = gr.Button("🗑️ Effacer")

                with gr.Accordion("📊 Métriques détaillées", open=True):
                    stats_text = gr.Markdown(
                        "**Statistiques :** en attente d'entrée..."
                    )

            with gr.Column(scale=3):
                plot_output = gr.Plot(label="Visualisation de l'embedding")

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

        def submit_text(text: str, points_cache: list) -> tuple:
            if not text or not text.strip():
                return None, "⚠️ **Veuillez saisir un texte valide.**", points_cache
            return visualize_embedding(text, points_cache)

        def clear_text(points_cache: list) -> tuple:
            return "", None, "Entrez un nouveau texte pour générer un embedding.", []

        submit_btn.click(
            fn=submit_text,
            inputs=[text_input, points_state],
            outputs=[plot_output, stats_text, points_state]
        )

        clear_btn.click(
            fn=clear_text,
            inputs=[points_state],
            outputs=[text_input, plot_output, stats_text, points_state]
        )

        text_input.submit(
            fn=submit_text,
            inputs=[text_input, points_state],
            outputs=[plot_output, stats_text, points_state]
        )

    return demo


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================
demo = create_interface()

if __name__ == "__main__":
    # [P3] Gradio 6.0 : theme passé à launch()
    demo.launch(server_name="0.0.0.0", server_port=7860)
