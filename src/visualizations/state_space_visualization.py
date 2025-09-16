from src.visualizations import BaseVisualization
import plotly.express as px


class StateSpaceVis(BaseVisualization):

    def __init__(self, embedding_method, clustering_method, dim_reduction_method):


        super().__init__(embedding_method, clustering_method, dim_reduction_method)

    def plot(self):
        _, cluster_labels, dim_reductions = self.load_data()
        columns = dim_reductions.columns
        fig = px.scatter_3d(dim_reductions, x=columns[0], y=columns[1], z=columns[2], color=cluster_labels['Cluster Label'])
        fig.update_layout(
            title='3D Cluster Plot',
            scene=dict(
                xaxis_title=columns[0],
                yaxis_title=columns[1],
                zaxis_title=columns[2]
            )
        )

        return fig


