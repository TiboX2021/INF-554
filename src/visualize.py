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
) -> None:
    """Plot a numpy embedding.
    Note : because the PCA and TSNE models take a lot of time to be fitted, this function will
    randomly select a maximum of 1000 embeddings to plot.

    Params :
        - embeddings (EMBEDDING_COUNT, FEATURES) : matrix of vector embeddings
        - labels (EMBEDDING_COUNT) : list of labels for the embeddings
    """
    # Random extraction :
    if len(embeddings) > 1000:
        indices = np.random.choice(len(embeddings), 1000)
        embeddings = embeddings[indices]
        labels = np.array(labels)[indices]

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
