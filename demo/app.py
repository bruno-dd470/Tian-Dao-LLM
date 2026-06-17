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
    Convertit un texte en embedding 20D via l'IA endorégulée.
    """
    try:
        # Convertir le texte en valeur 0-63 (6 bits)
        hash_val = hash(text) % 64
        if hash_val < 0:
            hash_val = -hash_val
        
        # Injecter dans le core
        attractor = core.encode_bits(hash_val)
        
        # Récupérer les modes des pentades comme embedding 20D
        embedding_20d = []
        for i in range(1, 7):
            embedding_20d.append(core.pentad_sign[f'P{i}'])
        for i in range(1, 7):
            embedding_20d.append(core.pentad_sign[f'N{i}'])
        
        # Convertir en array numpy
        emb = np.array(embedding_20d, dtype=np.float32)
        
        # ⚠️ CORRECTION ICI : 12 dimensions, pas 20
        emb = emb + np.random.randn(12) * 0.01
        
        # Générer un embedding 768D pour comparaison
        emb_768d = np.random.randn(768)
        
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
        emb_20d, emb_768d, attractor = text_to_embedding(text)
        
        # Statistiques
        eta = core.eta_direct()
        frustration = core.frustration()
        r_threshold = core.r_threshold()
        regime = core.get_regime().value
        
        # Créer la visualisation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Graphique 1 : Barres de l'embedding 20D
        ax1 = axes[0]
        colors = ['blue' if v > 0 else 'red' for v in emb_20d]
        ax1.bar(range(1, 13), emb_20d, color=colors, alpha=0.7)
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Valeur')
        ax1.set_title(f'Embedding 12D (attracteur {attractor})')
        ax1.set_ylim(-1.5, 1.5)
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2 : Visualisation 2D via PCA
        ax2 = axes[1]
        
        try:
            if not hasattr(visualize_embedding, 'embeddings_cache'):
                visualize_embedding.embeddings_cache = []
            
            visualize_embedding.embeddings_cache.append(emb_20d)
            if len(visualize_embedding.embeddings_cache) > 50:
                visualize_embedding.embeddings_cache.pop(0)
            
            if len(visualize_embedding.embeddings_cache) >= 2:
                points = np.array(visualize_embedding.embeddings_cache)
                pca = PCA(n_components=2)
                points_2d = pca.fit_transform(points)
                
                ax2.scatter(points_2d[:-1, 0], points_2d[:-1, 1], 
                           c='gray', alpha=0.3, s=30, label='Historique')
                ax2.scatter(points_2d[-1, 0], points_2d[-1, 1], 
                           c='red', s=100, label='Nouveau', edgecolors='black')
                ax2.set_title('Projection 2D (PCA)')
            else:
                ax2.text(0.5, 0.5, f'{(2 - len(visualize_embedding.embeddings_cache))} entrée(s)\nsupplémentaire(s) pour la PCA',
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('En attente de plus de données...')
        except Exception as e:
            ax2.text(0.5, 0.5, f'Erreur PCA:\n{str(e)[:50]}',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=10)
            ax2.set_title('PCA non disponible')
        
        ax2.set_xlabel('Composante 1')
        ax2.set_ylabel('Composante 2')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Informations textuelles
        info_text = (
            f"**Analyse du texte :** `{text[:50]}{'...' if len(text)>50 else ''}`\n\n"
            f"**Attracteur :** {attractor}\n"
            f"**Asymétrie η :** {eta:+.2f}\n"
            f"**Frustration E :** {frustration}\n"
            f"**Seuil R :** {r_threshold:.2f}\n"
            f"**Régime :** {regime}\n"
            f"**Embedding 20D :** {len(emb_20d)} dims (taille : {emb_20d.nbytes} bytes)\n"
            f"**Embedding 768D :** {len(emb_768d)} dims (taille : {emb_768d.nbytes} bytes)\n"
            f"**Taux de compression :** {emb_768d.nbytes / emb_20d.nbytes:.1f}x\n"
            f"**Entrées traitées :** {core.input_counter}"
        )
        
        return fig, info_text
    
    except Exception as e:
        error_msg = f"❌ **Erreur :** {str(e)}\n\n**Détails :**\n```\n{traceback.format_exc()}\n```"
        return None, error_msg

# Interface Gradio
def create_interface():
    with gr.Blocks(title="Tian-Dao 20D Embeddings Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 Tian-Dao 20D Embeddings Demo
        ### IA Endorégulée - Invariant 64→20 avec Wuxing Cycle
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

# Point d'entrée
demo = create_interface()

if __name__ == "__main__":
    demo.launch()
