import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os
import traceback

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer votre IA endorégulée
from Endoregulated_AI_v26 import EndoRegulatedCore, RandomInputSimulator

# Initialiser le core une fois (global)
core = EndoRegulatedCore(noise_level=0.15)

def text_to_embedding(text):
    """
    Convertit un texte en embedding 12D unique.
    Utilise le hash du texte pour moduler les pentades.
    """
    try:
        # Hash du texte pour obtenir des valeurs uniques
        hash_val = 0
        for i, char in enumerate(text):
            hash_val = (hash_val * 31 + ord(char)) % 64
        
        # Utiliser le hash pour créer un embedding unique
        # On module chaque pentade par le hash
        embedding = []
        
        # Pentades P1 à P6
        for i in range(1, 7):
            base_val = core.pentad_sign[f'P{i}']
            # Modulation par le hash et la position
            mod = 1 if (hash_val + i * 2) % 3 != 0 else -1
            embedding.append(base_val * mod)
        
        # Pentades N1 à N6
        for i in range(1, 7):
            base_val = core.pentad_sign[f'N{i}']
            mod = 1 if (hash_val + i * 3 + 1) % 3 != 0 else -1
            embedding.append(base_val * mod)
        
        # Convertir en array numpy
        emb = np.array(embedding, dtype=np.float32)
        
        # Ajouter un bruit dépendant du texte
        noise_seed = hash_val % 100
        np.random.seed(noise_seed)
        emb = emb + np.random.randn(12) * 0.15
        np.random.seed(None)
        
        # Normaliser pour rester dans [-1, 1]
        emb = np.clip(emb, -1.0, 1.0)
        
        # Générer un embedding 768D pour comparaison
        emb_768d = np.random.randn(768)
        
        # Injecter dans le core pour faire évoluer le système
        attractor = core.encode_bits(hash_val)
        
        return emb, emb_768d, attractor
    except Exception as e:
        print(f"Erreur dans text_to_embedding: {e}")
        traceback.print_exc()
        raise

def visualize_embedding(text):
    """
    Fonction principale de visualisation pour Gradio.
    """
    try:
        if not text or text.strip() == "":
            return None, "⚠️ **Veuillez entrer un texte valide.**"
        
        # Obtenir l'embedding
        emb_12d, emb_768d, attractor = text_to_embedding(text)
        
        # Statistiques
        eta = core.eta_direct()
        frustration = core.frustration()
        r_threshold = core.r_threshold()
        regime = core.get_regime().value
        
        # Créer la visualisation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Graphique 1 : Barres de l'embedding 12D
        ax1 = axes[0]
        colors = ['blue' if v > 0 else 'red' for v in emb_12d]
        bars = ax1.bar(range(1, len(emb_12d) + 1), emb_12d, color=colors, alpha=0.7)
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Valeur')
        ax1.set_title(f'Embedding 12D (attracteur {attractor})')
        ax1.set_ylim(-1.5, 1.5)
        ax1.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, emb_12d):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8)
        
        # Graphique 2 : Visualisation 2D via PCA
        ax2 = axes[1]
        
        if not hasattr(visualize_embedding, 'points_cache'):
            visualize_embedding.points_cache = []
        
        visualize_embedding.points_cache.append(emb_12d)
        if len(visualize_embedding.points_cache) > 20:
            visualize_embedding.points_cache.pop(0)
        
        if len(visualize_embedding.points_cache) >= 3:
            try:
                points = np.array(visualize_embedding.points_cache)
                pca = PCA(n_components=2)
                points_2d = pca.fit_transform(points)
                
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
        
        plt.close(fig)
        return fig, info_text
    
    except Exception as e:
        error_msg = f"❌ **Erreur :** {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return None, error_msg

# Interface Gradio
def create_interface():
    with gr.Blocks(title="Tian-Dao Embeddings Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 Tian-Dao Embeddings Demo
        ### IA Endorégulée - Invariant 64→12 avec Wuxing Cycle

        Cette démo transforme votre texte en un embedding 12D unique.
        Le système alterne entre les régimes **SHENG** (exploration) et **KE** (contraction).
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Texte à encoder",
                    placeholder="Saisissez votre texte ici...",
                    lines=3
                )
                submit_btn = gr.Button("🔮 Générer l'embedding", variant="primary")
                clear_btn = gr.Button("🗑️ Effacer")
                stats_text = gr.Markdown("**Statistiques :** en attente d'entrée...")
            
            with gr.Column(scale=3):
                plot_output = gr.Plot(label="Visualisation de l'embedding")
        
        gr.Examples(
            examples=[
                "Bonjour le monde",
                "L'intelligence artificielle est fascinante",
                "Le cycle Wuxing gouverne l'équilibre",
                "Sheng et Ke dansent dans le chaos",
                "La topologie des attracteurs révèle l'harmonie"
            ],
            inputs=text_input,
            label="📝 Exemples de textes"
        )
        
        def submit_text(text):
            if not text or text.strip() == "":
                return None, "⚠️ **Veuillez saisir un texte valide.**"
            return visualize_embedding(text)
        
        def clear_text():
            if hasattr(visualize_embedding, 'points_cache'):
                visualize_embedding.points_cache = []
            return "", None, "Entrez un nouveau texte pour générer un embedding."
        
        submit_btn.click(
            fn=submit_text,
            inputs=text_input,
            outputs=[plot_output, stats_text]
        )
        
        clear_btn.click(
            fn=clear_text,
            outputs=[text_input, plot_output, stats_text]
        )
        
        text_input.submit(
            fn=submit_text,
            inputs=text_input,
            outputs=[plot_output, stats_text]
        )
    
    return demo

demo = create_interface()

if __name__ == "__main__":
    demo.launch()
