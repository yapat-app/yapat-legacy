import plotly.express as px
import pandas as pd

from src.visualizations import BaseVisualization

class ClusterTimeGrid(BaseVisualization):

    def __init__(self, embedding_method, clustering_method, dim_reduction_method):

        super().__init__(embedding_method, clustering_method, dim_reduction_method)

    def plot(self):
        _, cluster_labels, _ = self.load_data()
        # apply method doesn't work on index directly, therefore using Series
        date_time_series = pd.Series(cluster_labels.index).apply(self.parse_datetime_from_filename)
        cluster_labels['date'], cluster_labels['time'] = zip(*date_time_series)
        cluster_labels['time_decimal'] = cluster_labels['time'].apply(lambda t: t.hour + t.minute / 60 + t.second / 3600)
        fig = px.scatter(cluster_labels, x='time_decimal', y='date', color='Cluster Label',
                         labels={
                             'time_decimal': 'Time (hours)',
                             'date': 'Date',
                             'cluster_label': 'Cluster'
                         },
                         title='Cluster Time Grid',
                         color_continuous_scale=px.colors.qualitative.Vivid)

        fig.update_layout(coloraxis_colorbar=dict(
            title='Cluster Label',
        ))
        fig.update_traces(marker=dict(size=10))
        fig.update_yaxes(autorange="reversed")
        return fig



