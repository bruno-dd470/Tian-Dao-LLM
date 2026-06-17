import gradio as gr
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_embedding(text):
    # Exemple : embeddings aléatoires (à remplacer par votre code)
    emb_20d = np.random.rand(20)
    emb_768d = np.random.rand(768)

    # Vérifier si on a assez de points pour la PCA (au moins 2)
    # Comme nous avons un seul embedding, on va créer un point de référence
    # ou simplement afficher l'embedding sans réduction
    try:
        # Pour un seul embedding, on ne peut pas faire de PCA en 2D
        # Solution : ajouter un point de référence (par exemple l'origine)
        points = np.array([emb_20d, np.zeros(20)])
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points)
        # Le premier point est notre embedding
        emb_20d_2d = points_2d[0:1]
    except ValueError:
        # En cas d'erreur, on affiche l'embedding sans réduction
        # On crée un graphique simple
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Embedding 20D:\n{emb_20d[:5]}...\n(affichage limité)", 
                ha='center', va='center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("20D Embedding (pas de réduction PCA disponible)")
        plt.close()
        return fig, f"20D: {emb_20d.nbytes} bytes | 768D: {emb_768d.nbytes} bytes"

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(emb_20d_2d[0, 0], emb_20d_2d[0, 1], label="20D", color="blue", s=100)
    ax.set_title("20D Embedding Projection (PCA 2D)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.close()

    return fig, f"20D: {emb_20d.nbytes} bytes | 768D: {emb_768d.nbytes} bytes"

iface = gr.Interface(
    fn=visualize_embedding,
    inputs=gr.Textbox(placeholder="Enter text to embed..."),
    outputs=[gr.Plot(label="Embedding Visualization"), gr.Textbox(label="Size Comparison")],
    title="Tian-Dao 20D Embeddings Demo"
)
iface.launch()
