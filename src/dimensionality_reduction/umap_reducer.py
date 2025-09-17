import pandas as pd
from src.dimensionality_reduction import BaseDimensionalityReduction

class UmapReducer(BaseDimensionalityReduction):

    def __init__(self, n_components: int = 3):
        """
        Initialize the Umap class with the specified number of components for UMAP.

        :param n_components: The number of components for UMAP.
        """
        super().__init__()
        self.n_components = n_components
        self.dim_reducer = None

    def fit_transform(self, embedding_method_name):
        if self.dim_reducer is None:
            from umap import UMAP  # Lazy import inside the function (it caused problems with file names etc)
            self.dim_reducer = UMAP(n_components=self.n_components)

        data = self.load_data(embedding_method_name)
        print('dataloaded')
        self.scaled_data = self.scale_data(data)
        print('data scaled')
        reduced_data = self.dim_reducer.fit_transform(self.scaled_data)
        # Creating a DataFrame for the reduced data
        columns = [f'UMAP {i + 1}' for i in range(self.n_components)]
        self.transformed_data = pd.DataFrame(reduced_data, columns=columns, index=data.index)
        print('data transformed')
        self.save_transformed_data('umap_reducer', embedding_method_name, self.transformed_data)
        print('data saved to db')
        return

