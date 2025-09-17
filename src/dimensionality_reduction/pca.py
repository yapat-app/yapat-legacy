import pandas as pd
from sklearn.decomposition import PCA as pca

from src.dimensionality_reduction import BaseDimensionalityReduction


class PCA(BaseDimensionalityReduction):
    """

    Attributes:
    -----------
    n_components : int
        The number of principal components to compute.

    Methods:
    --------
    fit_transform(data: pd.DataFrame) -> pd.DataFrame:
        Fit the PCA model to the data and transform it.
    """

    def __init__(self, n_components: int = 3):
        """
        Initialize the PCAReduction class.

        :param n_components: The number of components for PCA.
        """
        super().__init__()
        self.n_components = n_components
        self.dim_reducer = pca(n_components=self.n_components)

    def fit_transform(self, embedding_method_name):
        """
        Fit the PCA model to the dataset and transform the dataset.

        :param data: DataFrame containing the data for dimensionality reduction.
        :return: DataFrame with the reduced dimensions.
        """
        data = self.load_data(embedding_method_name)
        self.scaled_data = self.scale_data(data)
        reduced_data = self.dim_reducer.fit_transform(self.scaled_data)
        columns = [f'PC {i + 1}' for i in range(self.n_components)]
        self.transformed_data = pd.DataFrame(reduced_data, columns=columns, index=data.index)
        self.save_transformed_data('pca', embedding_method_name, self.transformed_data)
        return
