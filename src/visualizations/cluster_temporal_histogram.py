import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

from src.visualizations import BaseVisualization

class ClusterTemporalHist(BaseVisualization):

    def __init__(self, embedding_method, clustering_method, dim_reduction_method, average_based: str = 'hour'):

        self.average_based = average_based
        super().__init__(embedding_method, clustering_method, dim_reduction_method)



    def plot(self):
        _, cluster_labels, _ = self.load_data()
        # apply method doesn't work on index directly, therefore using Series
        date_time_series = pd.Series(cluster_labels.index).apply(self.parse_datetime_from_filename)
        cluster_labels['date'], cluster_labels['time'] = zip(*date_time_series)
        cluster_labels['Datetime'] = cluster_labels.apply(lambda row: datetime.combine(row['date'], row['time']),
                                                          axis=1)
        cluster_labels.set_index('Datetime', inplace=True)
        cluster_labels.sort_index(inplace=True)
        group_map = {
            'hour': cluster_labels.index.hour,
            'date': cluster_labels.index.date,
            'month': cluster_labels.index.month
        }
        df_count = cluster_labels.groupby(['Cluster Label', group_map[self.average_based]]).size().reset_index(name='Count')
        df_count.columns = ['Cluster Label', self.average_based, 'Count']
        fig = px.line(df_count, x=self.average_based, y='Count', color='Cluster Label',
                      title='Number of Cluster Occurrences over Time',
                      labels={self.average_based: self.average_based.capitalize(), 'Count': 'Occurrences',
                              'Cluster Label': 'Cluster'}, line_shape='spline')
        fig.update_layout(xaxis_title=self.average_based.capitalize(), yaxis_title='Occurrences')
        return fig







