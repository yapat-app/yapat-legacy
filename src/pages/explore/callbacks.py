import os
import json
from dash import Input, Output, State, callback, callback_context, dcc, html

from src.utils import get_embedding_model
from src.clustering import get_clustering_model
from src.dimensionality_reduction import get_dr_model
from src.evaluations.embedding_evaluation import EmbeddingsEvaluation
from src.evaluations.clustering_evaluation import ClusteringEvaluation
from src.visualizations import BaseVisualization
from src.visualizations.cluster_temporal_histogram import ClusterTemporalHist
from src.visualizations.cluster_time_grid import ClusterTimeGrid
from src.visualizations.state_space_visualization import StateSpaceVis
from src.visualizations.time_series_plot import TimeSeries
from src.visualizations.rose_plot import RosePlot
from dash import Input, Output, State, callback, callback_context

from src.extensions import sqlalchemy_db
from src.schema_model import (
    ClusteringMethod, ClusteringResult, 
    EmbeddingMethod, EmbeddingResult,
    DimReductionMethod, Dataset
)

pipeline_steps = {
    'embeddings': EmbeddingMethod,
    'clustering': ClusteringMethod,
    'dimensionality_reduction': DimReductionMethod
}

data = [
    {"metric": "F1 Score (Time Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
    {"metric": "Accuracy (Time Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
    {"metric": "F1 Score (Location Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
    {"metric": "Accuracy (Location Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
    {"metric": "Explained Variance", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
    {"metric": "Entropy", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
    {"metric": "Silhouette Index", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
    {"metric": "Davies Bouldin Index", "birdnet": 0, "acoustic_indices": 0, "vae": 0}
]

def update_db_methods():
    add_methods = []
    del_methods = []

    for package_name in pipeline_steps.keys():
        method_names = [file[:-3] for file in os.listdir(package_name) if
                        file.endswith('.py') and file != '__init__.py']
        existing_methods = list_existing_methods(package_name)
        method_names = set(method_names) - set(existing_methods)
        Table = pipeline_steps[package_name]
        add_methods += [Table(method_name=method_name) for method_name in method_names]

    return add_methods


def list_existing_methods(package_name):
    Table = pipeline_steps[package_name]
    existing_methods = sqlalchemy_db.session.execute(sqlalchemy_db.select(Table.method_name)).fetchall()
    existing_methods = [i[0] for i in existing_methods]
    return existing_methods


def list_existing_datasets():
    existing_datasets = sqlalchemy_db.session.execute(sqlalchemy_db.select(Dataset.dataset_name)).fetchall()
    existing_datasets = [i[0] for i in existing_datasets]
    return existing_datasets

def fetch_embedding_and_clustering_methods():

    embedding_methods = sqlalchemy_db.session.query(EmbeddingMethod.method_name).all()
    clustering_methods = sqlalchemy_db.session.query(ClusteringMethod.method_name).all()
    return embedding_methods, clustering_methods

def extract_evaluation_results():
    selected_dataset = sqlalchemy_db.session.query(Dataset).filter_by(is_selected=True).first()
    dataset_id = selected_dataset.id
    embedding_results = sqlalchemy_db.session.query(EmbeddingResult).filter_by(dataset_id=dataset_id, task='completed').all()
    for result in embedding_results:
        method = sqlalchemy_db.session.query(EmbeddingMethod).filter_by(id=result.embedding_id).first()
        if result.evaluation_results and result.evaluation_results != "{}":
            evaluation_data = json.loads(result.evaluation_results)
        else:
            evaluation_data = {}
        evaluation_data = json.loads(result.evaluation_results)
        if method.method_name in ["birdnet", "acoustic_indices", "vae"]:
            prefix = method.method_name
            data[0][prefix] = evaluation_data.get("f1_score_time", "N/A")
            data[1][prefix] = evaluation_data.get("accuracy_time", "N/A")
            data[2][prefix] = evaluation_data.get("f1_score_location", "N/A")
            data[3][prefix] = evaluation_data.get("accuracy_location", "N/A")
            data[4][prefix] = evaluation_data.get("Explained Variance", "N/A")
            data[5][prefix] = evaluation_data.get("Entropy", "N/A")
            data[6][prefix] = evaluation_data.get("Silhouette Index", "N/A")
            data[7][prefix] = evaluation_data.get("Davies Bouldin Index", "N/A")


    for result in embedding_results:
        method = sqlalchemy_db.session.query(EmbeddingMethod).filter_by(id=result.embedding_id).first()
        full_sil = []
        full_db = []
        completed_clustering_results = sqlalchemy_db.session.query(ClusteringResult).filter_by(
            task='completed', embedding_id=result.id).all()
        for clustering_result in completed_clustering_results:
            clustering_method = sqlalchemy_db.session.query(ClusteringMethod).filter_by(method_id=clustering_result.method_id).first()
            if clustering_result.evaluation_results and clustering_result.evaluation_results != "{}":
                evaluation_data = json.loads(clustering_result.evaluation_results)
            else:
                evaluation_data = {}
            full_sil.append(f"{clustering_method.method_name}:{evaluation_data.get('Silhouette Score', 'N/A')}")
            full_db.append(f"{clustering_method.method_name}:{evaluation_data.get('Davies Bouldin Score', 'N/A')}")
        data[6][method.method_name] = ", ".join(full_sil)
        data[7][method.method_name] = ", ".join(full_db)

    return data

@callback(
    Output("new-modal", "is_open"),
    [Input("new-confirm", "n_clicks"),
     Input("new-cancel", "n_clicks"),
     Input("new-pipeline", "n_clicks")],
    [State("new-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_new_modal(n_new, n_cancel, n_create, is_open):
    triggered_id = callback_context.triggered_id
    print(triggered_id)
    if triggered_id == 'new-pipeline':
        return not is_open
    else:
        return is_open

@callback(
    Output("load-modal", "is_open"),
    [Input("load-confirm", "n_clicks"),
     Input("load-cancel", "n_clicks"),
     Input("load-pipeline", "n_clicks")],
    [State("load-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_load_modal(n_load, n_cancel, n_confirm, is_open):
    triggered_id = callback_context.triggered_id
    print(triggered_id)
    if triggered_id == 'load-pipeline':
        return not is_open
    else:
        return is_open


# @callback(
#     Output("new-pipeline-summary", "children"),
#     Input("methods-embedding", "value"),
#     Input("methods-clustering", "value"),
#     Input("methods-dimred-viz", "value")
# )
# def display_pipeline_summary(m_e, m_c, m_dv):
#     m_e = m_e if type(m_e) == list else [m_e]
#     m_c = m_c if type(m_c) == list else [m_c]
#     m_dv = m_dv if type(m_dv) == list else [m_dv]
#     n_pipelines = len(m_e) * len(m_c) * len(m_dv)
#     msg = f"{n_pipelines} pipelines will be computed"
#     if n_pipelines == 1: msg = msg.replace("pipelines", "pipeline")
#     return msg


@callback(
    Output("evaluation-table", "data"),
    Input("new-confirm", "n_clicks"),
    State("new-methods-embedding", "value"),
    State("new-methods-clustering", "value"),
    State("new-methods-dimred-viz", "value"),
    prevent_initial_call=True
)

def run_pipeline(n_clicks, m_e, m_c, m_dv):
    if n_clicks is None:
        return
    if m_e:
        embedding_method = m_e[0] if isinstance(m_e, list) else m_e
        embedding_instance = get_embedding_model(embedding_method)
        embedding_instance.process()
        evaluation_instance = EmbeddingsEvaluation(embedding_method, None)
        evaluation_instance.evaluate()
        data = extract_evaluation_results()
        if m_c:
            clustering_method = m_c[0] if isinstance(m_c, list) else m_c
            clustering_instance = get_clustering_model(clustering_method)
            clustering_instance.fit(embedding_method)
            evaluation_instance = ClusteringEvaluation(embedding_method, clustering_method)
            evaluation_instance.evaluate()
            data = extract_evaluation_results()
        if m_dv:
            dim_reduction_method = m_dv[0] if isinstance(m_dv, list) else m_dv
            dim_reduction_instance = get_dr_model(dim_reduction_method)
            dim_reduction_instance.fit_transform(embedding_method)
        return data


@callback(
    Output('loaded-figures-store', 'data'),
    Output('status-box', 'children'),
    Input('load-confirm', 'n_clicks'),
    State("load-methods-embedding", "value"),
    State("load-methods-clustering", "value"),
    State("load-methods-dimred-viz", "value"),
    prevent_initial_call=True
)
def load_figures(n_clicks, m_e, m_c, m_dv):
    if n_clicks > 0:
        embedding_method = m_e[0] if isinstance(m_e, list) else m_e
        clustering_method = m_c[0] if isinstance(m_c, list) else m_c
        dim_reduction_method = m_dv[0] if isinstance(m_dv, list) else m_dv

        figures = {
            "cluster-time-histogram": ClusterTemporalHist(embedding_method, clustering_method,
                                                          dim_reduction_method).plot(),
            "cluster-time-grid": ClusterTimeGrid(embedding_method, clustering_method, dim_reduction_method).plot(),
            "time-series": TimeSeries(embedding_method, clustering_method, dim_reduction_method).plot(),
            "temp-rose-plot": RosePlot(embedding_method, clustering_method, dim_reduction_method).plot(),
            "cluster-state-space": StateSpaceVis(embedding_method, clustering_method, dim_reduction_method).plot()
        }
        return figures, "Figures Fetched. Please view them in the respective Tabs"
    return {}, "No figures are loaded."

@callback(
    Output('visualization-content', 'children'),
    Output('status-box', 'children', allow_duplicate=True),
    [Input('visualization-tabs', 'active_tab')],
    [State('loaded-figures-store', 'data')],
    prevent_initial_call=True
)
def update_visualization_content(active_tab, figures_data):
    # If no figures have been loaded, prompt the user
    if not figures_data:
        return "", "Please load a pipeline to view visualizations."
    figure_data = figures_data.get(active_tab)
    if figure_data:
        figure_component = dcc.Graph(figure=figure_data)
    else:
        figure_component = html.Div("Figure not found for this tab.")

    return figure_component, ""







@callback(
    Output("create-pipeline-msg", "children"),
    Input("create-pipeline", "n_clicks"),
    State("project-content", "data"),
    State("methods-embedding", "value"),
    State("methods-clustering", "value"),
    prevent_initial_call=True
)
def process_pipeline_create_click(n_clicks, project_content, list_embedding_methods, list_clustering_methods):
    dataset_name = project_content.get("project_name")
    list_embedding_methods = list_embedding_methods if isinstance(list_embedding_methods, list) else [
        list_embedding_methods]
    list_clustering_methods = list_clustering_methods if isinstance(list_clustering_methods, list) else [
        list_clustering_methods]
    if callback_context.triggered_id == "create-pipeline":
        from src.utils.task_manager import compute_clusters, compute_embeddings
        compute_embeddings(dataset_name=dataset_name, list_embedding_methods=list_embedding_methods)
        compute_clusters(dataset_name=dataset_name, list_clustering_methods=list_clustering_methods)

    return f"Created {n_clicks} pipelines"
