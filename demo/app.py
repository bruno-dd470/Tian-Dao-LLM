import gradio as gr
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_embedding(text):
    # Exemple : embeddings aléatoires (à remplacer par votre code)
    emb_20d = np.random.rand(20)
    emb_768d = np.random.rand(768)

    # Réduction pour visualisation
    pca = PCA(n_components=2)
    emb_20d_2d = pca.fit_transform(emb_20d.reshape(1, -1))

    # Plot
    fig, ax = plt.subplots()
    ax.scatter(emb_20d_2d[0, 0], emb_20d_2d[0, 1], label="20D", color="blue")
    ax.set_title("20D Embedding Projection")
    ax.legend()
    plt.close()

    return fig, f"20D: {emb_20d.nbytes} bytes | 768D: {emb_768d.nbytes} bytes"

iface = gr.Interface(
    fn=visualize_embedding,
    inputs=gr.Textbox(placeholder="Enter text to embed..."),
    outputs=[gr.Plot(label="Embedding Visualization"), gr.Textbox(label="Size Comparison")],
    title="Tian-Dao 20D Embeddings Demo"
)
iface.launch()
