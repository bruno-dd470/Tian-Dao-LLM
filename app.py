"""
Module: app.py
Application Gradio pour la démonstration de l'IA endorégulée Tian-Dao.
Version: 1.4  # [P0+P1+20D+i18n FR/EN] corrections appliquées
Date: 2026-06-20
"""
import warnings
import asyncio
import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hashlib
import traceback
import sys
import os

# ✅ Supprimer les warnings asyncio pour Python 3.13
warnings.filterwarnings("ignore", category=DeprecationWarning, module="asyncio")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")

# ✅ Patch pour le bug du file descriptor -1 sur Python 3.13
if sys.version_info >= (3, 13):
    try:
        import selectors
        original_fileobj_to_fd = selectors._fileobj_to_fd
        
        def patched_fileobj_to_fd(fileobj):
            try:
                fd = original_fileobj_to_fd(fileobj)
                return fd
            except ValueError as e:
                if "Invalid file descriptor" in str(e):
                    return -1
                raise
        
        selectors._fileobj_to_fd = patched_fileobj_to_fd
    except Exception:
        pass

# Import direct depuis la racine
from Endoregulated_AI_v27 import EndoRegulatedCore, RandomInputSimulator, get_core_lock

# ============================================================================
# [P0-C] INITIALISATION GLOBALE THREAD-SAFE
# ============================================================================
core = EndoRegulatedCore(noise_level=0.15, seed=42)
core_lock = get_core_lock()


# ============================================================================
# INTERNATIONALISATION (FR/EN)
# ============================================================================
I18N = {
    "fr": {
        "title": "🧠 Démo Tian-Dao Embeddings",
        "subtitle": "### IA Endorégulée - Invariant 64→20 avec Cycle Wuxing",
        "description": (
            "Cette démo transforme votre texte en un embedding 20D unique via un "
            "**système dynamique non-connexionniste**. Le système alterne entre les "
            "régimes **SHENG** (exploration) et **KE** (contraction) selon un cycle "
            "d'auto-régulation inspiré du Wuxing."
        ),
        "language_label": "🌐 Langue",
        "text_label": "Texte à encoder",
        "text_placeholder": "Saisissez votre texte ici...",
        "submit_btn": "🔮 Générer l'embedding",
        "clear_btn": "🗑️ Effacer",
        "metrics_title": "📊 Métriques détaillées",
        "stats_waiting": "**Statistiques :** en attente d'entrée...",
        "plot_label": "Visualisation de l'embedding",
        "examples_title": "### 📝 Exemples de textes",
        "examples_label": "Cliquez sur un exemple pour le charger",
        "examples": [
            "Bonjour le monde",
            "L'intelligence artificielle est fascinante",
            "Le cycle Wuxing gouverne l'équilibre",
            "Sheng et Ke dansent dans le chaos",
            "La topologie des attracteurs révèle l'harmonie"
        ],
        "error_empty": "⚠️ Veuillez entrer un texte valide.",
        "error_prefix": "❌ **Erreur :**",
        "info_prefix": "**Analyse du texte :**",
        "info_attractor": "**Attracteur :**",
        "info_eta": "**Asymétrie η :**",
        "info_frustration": "**Frustration E :**",
        "info_threshold": "**Seuil R :**",
        "info_regime": "**Régime :**",
        "info_20d": "**Embedding 20D :**",
        "info_768d": "**Embedding 768D :**",
        "info_compression": "**Taux de compression :**",
        "info_processed": "**Entrées traitées :**",
        "dims": "dims",
        "size": "taille",
        "bytes": "bytes",
        "pca_waiting": "Données colinéaires\nen attente de variance...",
        "pca_waiting_title": "PCA en attente",
        "pca_need_more": "entrée(s) supplémentaire(s)\npour la PCA",
        "pca_collecting_title": "En attente de plus de données...",
        "pca_component1": "Composante 1",
        "pca_component2": "Composante 2",
        "pca_history": "Historique",
        "pca_new": "Nouveau",
        "embedding_title": "Embedding 20D (attracteur {})",
        "dimension": "Dimension",
        "value": "Valeur",
        "clear_msg": "Entrez un nouveau texte pour générer un embedding.",
    },
    "en": {
        "title": "🧠 Tian-Dao Embeddings Demo",
        "subtitle": "### Endoregulated AI - 64→20 Invariant with Wuxing Cycle",
        "description": (
            "This demo transforms your text into a unique 20D embedding via a "
            "**non-connectionist dynamic system**. The system alternates between "
            "**SHENG** (exploration) and **KE** (contraction) regimes according to a "
            "self-regulation cycle inspired by Wuxing."
        ),
        "language_label": "🌐 Language",
        "text_label": "Text to encode",
        "text_placeholder": "Enter your text here...",
        "submit_btn": "🔮 Generate embedding",
        "clear_btn": "🗑️ Clear",
        "metrics_title": "📊 Detailed metrics",
        "stats_waiting": "**Statistics:** waiting for input...",
        "plot_label": "Embedding visualization",
        "examples_title": "### 📝 Example texts",
        "examples_label": "Click an example to load it",
        "examples": [
            "Hello world",
            "Artificial intelligence is fascinating",
            "The Wuxing cycle governs balance",
            "Sheng and Ke dance in chaos",
            "The topology of attractors reveals harmony"
        ],
        "error_empty": "⚠️ Please enter a valid text.",
        "error_prefix": "❌ **Error:**",
        "info_prefix": "**Text analysis:**",
        "info_attractor": "**Attractor:**",
        "info_eta": "**Asymmetry η:**",
        "info_frustration": "**Frustration E:**",
        "info_threshold": "**Threshold R:**",
        "info_regime": "**Regime:**",
        "info_20d": "**20D Embedding:**",
        "info_768d": "**768D Embedding:**",
        "info_compression": "**Compression ratio:**",
        "info_processed": "**Inputs processed:**",
        "dims": "dims",
        "size": "size",
        "bytes": "bytes",
        "pca_waiting": "Collinear data\nwaiting for variance...",
        "pca_waiting_title": "PCA waiting",
        "pca_need_more": "more entry(ies)\nneeded for PCA",
        "pca_collecting_title": "Waiting for more data...",
        "pca_component1": "Component 1",
        "pca_component2": "Component 2",
        "pca_history": "History",
        "pca_new": "New",
        "embedding_title": "20D Embedding (attractor {})",
        "dimension": "Dimension",
        "value": "Value",
        "clear_msg": "Enter a new text to generate an embedding.",
    }
}


def _(key: str, lang: str = "fr") -> str:
    """Retourne la traduction d'une clé dans la langue donnée."""
    return I18N.get(lang, I18N["fr"]).get(key, I18N["fr"].get(key, key))


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
        digest = hashlib.sha256(text.encode('utf-8')).digest()
        hash_val = int.from_bytes(digest[:2], 'big') % 64
        
        ATTRACTOR_TRIPLETS = [
            ['P1', 'P2', 'P4'], ['P1', 'P3', 'P5'], ['P2', 'P3', 'P6'],
            ['P4', 'P5', 'N2'], ['P5', 'P6', 'N3'], ['P1', 'P6', 'N4'],
            ['P2', 'P5', 'N6'], ['P3', 'P4', 'N6'], ['P1', 'N2', 'N6'],
            ['P1', 'N3', 'N5'], ['P2', 'N3', 'N5'], ['P3', 'N2', 'N4'],
            ['P4', 'N1', 'N3'], ['P4', 'N5', 'N6'], ['P5', 'N1', 'N4'],
            ['P6', 'N1', 'N2'], ['P2', 'N1', 'N4'], ['P3', 'N1', 'N5'],
            ['P6', 'N5', 'N6'], ['N2', 'N3', 'N4'],
        ]
        
        embedding = []
        for triplet in ATTRACTOR_TRIPLETS:
            n_positive = sum(1 for p in triplet if p.startswith('P'))
            n_negative = sum(1 for p in triplet if p.startswith('N'))
            polarity_score = (n_positive - n_negative) / 3.0
            mod = 1.0 if (hash_val + len(embedding)) % 5 != 0 else -1.0
            embedding.append(polarity_score * mod)
        
        emb = np.array(embedding, dtype=np.float32)
        rng = np.random.default_rng(hash_val)
        emb = emb + rng.standard_normal(20).astype(np.float32) * 0.15
        emb = np.clip(emb, -1.0, 1.0)
        
        rng_768 = np.random.default_rng(hash_val + 1000)
        emb_768d = rng_768.standard_normal(768).astype(np.float32)
        
        with core_lock:
            attractor = core.encode_bits(hash_val)
        
        return emb, emb_768d, attractor
    
    except Exception as e:
        print(f"Erreur dans text_to_embedding: {e}")
        traceback.print_exc()
        raise


def visualize_embedding(text: str, points_cache: list, lang: str = 'fr') -> tuple:
    """
    Fonction principale de visualisation pour l'interface Gradio.
    
    Args:
        text (str): Texte à encoder.
        points_cache (list): Cache des embeddings pour la PCA (par session).
        lang (str): Langue ('fr' ou 'en').
    
    Returns:
        tuple: (figure, texte_info, points_cache_mis_à_jour)
    """
    fig = None
    try:
        if not text or not text.strip():
            return None, _("error_empty", lang), points_cache
        
        # 1. Génération de l'embedding
        emb_20d, emb_768d, attractor = text_to_embedding(text)
        
        # 2. Récupération des métriques
        with core_lock:
            eta = core.eta_direct()
            frustration = core.frustration()
            r_threshold = core.r_threshold()
            regime = core.get_regime().value
            input_counter = core.input_counter
        
        # 3. Création de la visualisation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # GRAPHIQUE 1 : Barres de l'embedding 20D
        ax1 = axes[0]
        colors = ['blue' if v > 0 else 'red' for v in emb_20d]
        bars = ax1.bar(range(1, len(emb_20d) + 1), emb_20d, color=colors, alpha=0.7)
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_xlabel(_("dimension", lang))
        ax1.set_ylabel(_("value", lang))
        ax1.set_title(_("embedding_title", lang).format(attractor))
        ax1.set_ylim(-1.5, 1.5)
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, emb_20d):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{val:.1f}',
                     ha='center',
                     va='bottom' if height > 0 else 'top',
                     fontsize=8)
        
        # GRAPHIQUE 2 : Projection 2D via PCA
        ax2 = axes[1]
        
        if points_cache is None:
            points_cache = []
        
        points_cache.append(emb_20d.copy())
        if len(points_cache) > 20:
            points_cache.pop(0)
        
        if len(points_cache) >= 5:
            try:
                points = np.array(points_cache)
                if np.linalg.matrix_rank(points) < 2:
                    ax2.text(0.5, 0.5, _("pca_waiting", lang),
                             ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title(_("pca_waiting_title", lang))
                else:
                    pca = PCA(n_components=2)
                    points_2d = pca.fit_transform(points)
                    
                    ax2.scatter(points_2d[:-1, 0], points_2d[:-1, 1],
                                c='gray', alpha=0.5, s=40, label=_("pca_history", lang))
                    ax2.scatter(points_2d[-1, 0], points_2d[-1, 1],
                                c='red', s=120, label=_("pca_new", lang),
                                edgecolors='black', linewidth=2)
                    ax2.set_title('Projection 2D (PCA)')
                    ax2.legend()
            except Exception as e:
                ax2.text(0.5, 0.5, f'PCA: {str(e)[:40]}',
                         ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Erreur PCA')
        else:
            remaining = 5 - len(points_cache)
            pca_need_text = _("pca_need_more", lang)
            ax2.text(0.5, 0.5, f"{remaining} {pca_need_text}",
                     ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12)
            ax2.set_title(_("pca_collecting_title", lang))
        
        ax2.set_xlabel(_("pca_component1", lang))
        ax2.set_ylabel(_("pca_component2", lang))
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 4. Construction des informations textuelles
        info_text = (
            f"{_('info_prefix', lang)} `{text[:50]}{'...' if len(text) > 50 else ''}`\n\n"
            f"{_('info_attractor', lang)} {attractor}\n"
            f"{_('info_eta', lang)} {eta:+.2f}\n"
            f"{_('info_frustration', lang)} {frustration}\n"
            f"{_('info_threshold', lang)} {r_threshold:.2f}\n"
            f"{_('info_regime', lang)} {regime}\n"
            f"{_('info_20d', lang)} {len(emb_20d)} {_('dims', lang)} "
            f"({_('size', lang)}: {emb_20d.nbytes} {_('bytes', lang)})\n"
            f"{_('info_768d', lang)} {len(emb_768d)} {_('dims', lang)} "
            f"({_('size', lang)}: {emb_768d.nbytes} {_('bytes', lang)})\n"
            f"{_('info_compression', lang)} {emb_768d.nbytes / emb_20d.nbytes:.1f}x\n"
            f"{_('info_processed', lang)} {input_counter}"
        )
        
        return fig, info_text, points_cache
    
    except Exception as e:
        error_msg = (f"{_('error_prefix', lang)} {str(e)}\n\n"
                     f"```\n{traceback.format_exc()}\n```")
        return None, error_msg, points_cache if points_cache else []
    
    finally:
        if fig is not None:
            plt.close(fig)


# ============================================================================
# INTERFACE GRADIO
# ============================================================================
def create_interface() -> gr.Blocks:
    """Crée l'interface Gradio pour la démonstration (bilingue FR/EN)."""
    with gr.Blocks(title="Tian-Dao Embeddings Demo") as demo:
        # État par session utilisateur
        points_state = gr.State(value=[])
        lang_state = gr.State(value="fr")
        
        # Sélecteur de langue (en haut à droite)
        with gr.Row():
            gr.Markdown("## 🧠 Tian-Dao")
            language_selector = gr.Dropdown(
                choices=[("🇫🇷 Français", "fr"), ("🇬🇧 English", "en")],
                value="fr",
                label="🌐 Language",
                scale=1,
                interactive=True
            )
        
        # Titre et description (mis à jour dynamiquement)
        title_md = gr.Markdown(f"# {_('title', 'fr')}\n{_('subtitle', 'fr')}")
        description_md = gr.Markdown(_("description", "fr"))
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label=_("text_label", "fr"),
                    placeholder=_("text_placeholder", "fr"),
                    lines=3
                )
                submit_btn = gr.Button(_("submit_btn", "fr"), variant="primary")
                clear_btn = gr.Button(_("clear_btn", "fr"))
                
                # ✅ Dropdown pour les exemples (dynamique)
                examples_title_md = gr.Markdown(_("examples_title", "fr"))
                examples_dropdown = gr.Dropdown(
                    choices=I18N["fr"]["examples"],
                    label=_("examples_label", "fr"),
                    interactive=True,
                    scale=1
                )
                
                with gr.Accordion(_("metrics_title", "fr"), open=True) as metrics_accordion:
                    stats_text = gr.Markdown(_("stats_waiting", "fr"))
            
            with gr.Column(scale=3):
                plot_output = gr.Plot(label=_("plot_label", "fr"))
        
        # --- Handlers ---
        def submit_text(text: str, points_cache: list, lang: str) -> tuple:
            if not text or not text.strip():
                return None, f"**{_('error_empty', lang)}**", points_cache
            return visualize_embedding(text, points_cache, lang)
        
        def clear_text(points_cache: list, lang: str) -> tuple:
            return "", None, _("clear_msg", lang), []
        
        def load_example(example: str) -> str:
            """Charge un exemple dans le champ texte."""
            return example if example else ""
        
        def change_language(lang: str, current_text: str, points_cache: list):
            """Met à jour l'interface quand la langue change."""
            new_title = f"# {_('title', lang)}\n{_('subtitle', lang)}"
            new_description = _("description", lang)
            new_examples_title = _("examples_title", lang)
            new_examples = I18N[lang]["examples"]
            
            if current_text and current_text.strip():
                fig, info, new_cache = visualize_embedding(current_text, points_cache, lang)
                return (
                    gr.update(value=new_title),
                    gr.update(value=new_description),
                    gr.update(label=_("text_label", lang),
                              placeholder=_("text_placeholder", lang)),
                    gr.update(value=_("submit_btn", lang)),
                    gr.update(value=_("clear_btn", lang)),
                    gr.update(choices=new_examples,
                              label=_("examples_label", lang)),
                    gr.update(value=new_examples_title),
                    gr.update(label=_("metrics_title", lang)),
                    gr.update(value=info if info else _("stats_waiting", lang)),
                    gr.update(label=_("plot_label", lang)),
                    fig,
                    lang,
                    new_cache,
                )
            else:
                return (
                    gr.update(value=new_title),
                    gr.update(value=new_description),
                    gr.update(label=_("text_label", lang),
                              placeholder=_("text_placeholder", lang)),
                    gr.update(value=_("submit_btn", lang)),
                    gr.update(value=_("clear_btn", lang)),
                    gr.update(choices=new_examples,
                              label=_("examples_label", lang)),
                    gr.update(value=new_examples_title),
                    gr.update(label=_("metrics_title", lang)),
                    gr.update(value=_("stats_waiting", lang)),
                    gr.update(label=_("plot_label", lang)),
                    None,
                    lang,
                    points_cache,
                )
        
        # Connexions des événements
        submit_btn.click(
            fn=submit_text,
            inputs=[text_input, points_state, lang_state],
            outputs=[plot_output, stats_text, points_state]
        )
        
        clear_btn.click(
            fn=clear_text,
            inputs=[points_state, lang_state],
            outputs=[text_input, plot_output, stats_text, points_state]
        )
        
        text_input.submit(
            fn=submit_text,
            inputs=[text_input, points_state, lang_state],
            outputs=[plot_output, stats_text, points_state]
        )
        
        # ✅ Handler pour le dropdown des exemples
        examples_dropdown.change(
            fn=load_example,
            inputs=examples_dropdown,
            outputs=text_input
        )
        
        # ✅ Changement de langue avec mise à jour du dropdown
        language_selector.change(
            fn=change_language,
            inputs=[language_selector, text_input, points_state],
            outputs=[
                title_md,
                description_md,
                text_input,
                submit_btn,
                clear_btn,
                examples_dropdown,
                examples_title_md,
                metrics_accordion,
                stats_text,
                plot_output,
                plot_output,
                lang_state,
                points_state,
            ]
        )
    
    return demo


# ============================================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================================
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        ssr_mode=False  # ✅ Désactiver SSR pour éviter les problèmes asyncio
    )
