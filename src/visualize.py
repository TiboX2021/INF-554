"""
Visualize embeddings using plots
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor


def detach_tensor(tensor: Tensor) -> np.ndarray:
    """Detach and convert a GPU tensor embedding to a numpy array"""
    return tensor.detach().cpu().numpy()


def plot_2D_embeddings(
    embeddings: np.ndarray, labels: list[str] | npt.NDArray[np.str_]
):
    """Plot a numpy embedding

    Params :
        - embeddings (EMBEDDING_COUNT, FEATURES) : matrix of vector embeddings
        - labels (EMBEDDING_COUNT) : list of labels for the embeddings
    """

    # Extract the 10 most significant dimensions from the embedding
    pca = PCA(n_components=10)
    # Compute a 2D projection of the 10D data that best represents the relations between embeddings
    tsne = TSNE(n_components=2)

    vecs_pca = pca.fit_transform(embeddings)
    vecs_tsne = tsne.fit_transform(vecs_pca)

    plt.figure(figsize=(7, 7))
    plt.title("TSNE visualisation of the GCN embeddings")
    colours = sns.color_palette("hls", len(np.unique(labels)))
    sns.scatterplot(
        x=vecs_tsne[:, 0],
        y=vecs_tsne[:, 1],
        hue=labels,
        legend="full",
        palette=colours,
    )

    plt.show()
