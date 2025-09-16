import pandas as pd
from datetime import datetime
import calendar
import plotly.express as px

from src.visualizations import BaseVisualization


class TimeSeries(BaseVisualization):
    def __init__(self, embedding_method, clustering_method, dim_reduction_method, resolution='Minutes', x_axis='linear', y_variables=None):
        super().__init__(embedding_method, clustering_method, dim_reduction_method)
        self.resolution = resolution  # Set the desired resolution (See create_numeric_time)
        self.x_axis = x_axis # Set the x-axis type ('year_cycle', 'diel_cycle', 'linear')
        self.y_variables = y_variables # A list of values to plot

    def create_numeric_time(self,df):
        """Create numeric time based on the resolution for cyclic plots."""
        if self.x_axis == 'diel_cycle':
            if self.resolution == 'Minutes':
                return df.index.hour + df.index.month/60
            else:
                return df.index.hour
        elif self.x_axis == 'year_cycle':
            if self.resolution == 'Monthly':
                    return df.index.month
            elif self.resolution == 'Daily':
                return df.index.month + df.index.day / 30
            elif self.resolution == 'Hourly':
                return df.index.month + df.index.day / 30 + df.index.hour / 720
            elif self.resolution == 'Minutes':
                return df.index.month + df.index.day / 30 + df.index.hour / 720 + df.index.minute / 43200

    def plot(self):
        embeddings, _, _ = self.load_data()
        if self.y_variables is None:
            self.y_variables = embeddings.columns[:3]
        y_variables_str = ', '.join(self.y_variables)
        date_time_series = pd.Series(embeddings.index).apply(self.parse_datetime_from_filename)
        embeddings['date'], embeddings['time'] = zip(*date_time_series)
        embeddings['Datetime'] = embeddings.apply(lambda row: datetime.combine(row['date'], row['time']),
                                                          axis=1)
        embeddings.set_index('Datetime', inplace=True)
        if self.x_axis == 'linear':
            fig = px.scatter(embeddings, x=embeddings.index, y=self.y_variables, trendline='lowess', opacity=0,
                             trendline_options=dict(frac=0.2), title=f'{y_variables_str} over Time with {self.resolution} resolution',
                             labels={'index': 'Date', 'y': f'Average {y_variables_str} Value'})
        else:
            # For cyclic time representations
            embeddings['Numeric Time'] = self.create_numeric_time(embeddings)
            ticktext = calendar.month_abbr[1:13] if self.x_axis == 'year_cycle' else [f"{h:02}:00" for h in range(24)]
            tickvals = list(range(1, 13)) if self.x_axis == 'year_cycle' else list(range(24))

            fig = px.scatter(embeddings, x='Numeric Time', y=self.y_variables, trendline='lowess', opacity=0,
                             trendline_options=dict(frac=0.2),
                             hover_data={'date': True, 'time':True},
                             labels={'x': 'Time', 'y': f'Average {self.y_variables} Value'})

            fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext)

        fig.update_layout(title=f'{y_variables_str} over Time with {self.resolution} resolution')
        return fig

