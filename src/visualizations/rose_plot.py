import plotly.express as px
import pandas as pd

from src.visualizations import BaseVisualization

class RosePlot(BaseVisualization):

    def __init__(self, embedding_method, clustering_method, dim_reduction_method):

        super().__init__(embedding_method, clustering_method, dim_reduction_method)

    def plot(self):
        _, cluster_labels, _ = self.load_data()
        # apply method doesn't work on index directly, therefore using Series
        date_time_series = pd.Series(cluster_labels.index).apply(self.parse_datetime_from_filename)
        cluster_labels['date'], cluster_labels['time'] = zip(*date_time_series)
        cluster_labels['time'] = pd.to_datetime(cluster_labels['time'], format='%H:%M:%S')
        cluster_labels['Hour'] = cluster_labels['time'].dt.hour
        df_count = cluster_labels.groupby(['Cluster Label', 'Hour']).size().reset_index(name='Count')
        df_count = df_count.assign(r=(df_count["Hour"] / 24) * 360)
        fig = px.bar_polar(df_count, r="Count", theta="r",
                           color='Cluster Label', hover_data={"Hour"}, template="plotly_dark",
                           color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig.update_layout(
            polar={
                "angularaxis": {
                    "tickmode": "array",
                    "tickvals": list(range(0, 360, 15)),
                    "ticktext": [f"{a:02}:00" for a in range(0, 24)],
                }
            },
            title='Cluster Rose Plot'
        )
        return fig


