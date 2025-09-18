import os
import json
import time
import dash
from dash import Input, Output, State, callback, callback_context, dcc, html, dash_table
import dash_bootstrap_components as dbc

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

# Simple cache for evaluation results
_evaluation_cache = {}
_cache_timestamp = 0
CACHE_DURATION = 30  # Cache for 30 seconds

def clear_evaluation_cache():
    """Clear the evaluation cache when new data is created"""
    global _evaluation_cache, _cache_timestamp
    _evaluation_cache = {}
    _cache_timestamp = 0

pipeline_steps = {
    'embeddings': EmbeddingMethod,
    'clustering': ClusteringMethod,
    'dimensionality_reduction': DimReductionMethod
}

# Global data structure moved to extract_evaluation_results function

def update_db_methods():
    add_methods = []
    del_methods = []

    for package_name in pipeline_steps.keys():
        # Prefer the package directory under src/ (e.g., src/embeddings) but
        # also tolerate deployments where the package is mounted at the repo root.
        package_path_src = os.path.join('src', package_name)
        method_names = []

        if os.path.isdir(package_path_src):
            try:
                method_names = [f[:-3] for f in os.listdir(package_path_src)
                                if f.endswith('.py') and f != '__init__.py']
            except OSError as e:
                # Listing failed (permission/IO) — log and continue with empty list
                print(f"Warning: could not list directory {package_path_src}: {e}")
                method_names = []
        elif os.path.isdir(package_name):
            try:
                method_names = [f[:-3] for f in os.listdir(package_name)
                                if f.endswith('.py') and f != '__init__.py']
            except OSError as e:
                print(f"Warning: could not list directory {package_name}: {e}")
                method_names = []
        else:
            # No directory found for this pipeline step; continue silently
            method_names = []

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
    # Declare global variables at the beginning
    global _evaluation_cache, _cache_timestamp
    
    # Check cache first
    current_time = time.time()
    if _evaluation_cache and (current_time - _cache_timestamp) < CACHE_DURATION:
        return _evaluation_cache
    
    # Reset data to initial state
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
    
    selected_dataset = sqlalchemy_db.session.query(Dataset).filter_by(is_selected=True).first()
    if not selected_dataset:
        return data
        
    dataset_id = selected_dataset.id
    
    # OPTIMIZATION: Single query with joins to get all data at once
    from sqlalchemy.orm import joinedload
    
    # Get embedding results with their methods in one query
    embedding_results = sqlalchemy_db.session.query(EmbeddingResult)\
        .join(EmbeddingMethod, EmbeddingResult.embedding_id == EmbeddingMethod.id)\
        .filter(EmbeddingResult.dataset_id == dataset_id, EmbeddingResult.task_state == 'completed')\
        .all()
    
    
    # Process embedding results
    for result in embedding_results:
        method = result.method  # Access via relationship instead of separate query
        if result.evaluation_results and result.evaluation_results != "{}":
            evaluation_data = json.loads(result.evaluation_results)
        else:
            evaluation_data = {}
            
        if method.method_name in ["birdnet", "acoustic_indices", "vae"]:
            prefix = method.method_name
            data[0][prefix] = evaluation_data.get("f1_score_time", "N/A")
            data[1][prefix] = evaluation_data.get("accuracy_time", "N/A")
            data[2][prefix] = evaluation_data.get("f1_score_location", "N/A")
            data[3][prefix] = evaluation_data.get("accuracy_location", "N/A")
            data[4][prefix] = evaluation_data.get("Explained Variance", "N/A")
            data[5][prefix] = evaluation_data.get("Entropy", "N/A")
            data[6][prefix] = evaluation_data.get("Silhouette Score", "N/A")
            data[7][prefix] = evaluation_data.get("Davies Bouldin Score", "N/A")

    # OPTIMIZATION: Get all clustering results with their methods in one query
    clustering_results = sqlalchemy_db.session.query(ClusteringResult)\
        .join(ClusteringMethod, ClusteringResult.method_id == ClusteringMethod.id)\
        .join(EmbeddingResult, ClusteringResult.embedding_result_id == EmbeddingResult.id)\
        .join(EmbeddingMethod, EmbeddingResult.embedding_id == EmbeddingMethod.id)\
        .filter(ClusteringResult.task_state == 'completed', EmbeddingResult.dataset_id == dataset_id)\
        .all()
    
    
    # Group clustering results by embedding method
    clustering_by_embedding = {}
    for clustering_result in clustering_results:
        embedding_method_name = clustering_result.embedding.method.method_name
        if embedding_method_name not in clustering_by_embedding:
            clustering_by_embedding[embedding_method_name] = []
        clustering_by_embedding[embedding_method_name].append(clustering_result)
    
    # Process clustering results
    for embedding_method_name, clustering_list in clustering_by_embedding.items():
        if embedding_method_name in ["birdnet", "acoustic_indices", "vae"]:
            # For clustering metrics, we'll show the best result or first available result
            best_sil = None
            best_db = None
            
            for clustering_result in clustering_list:
                clustering_method_name = clustering_result.method.method_name
                if clustering_result.evaluation_results and clustering_result.evaluation_results != "{}":
                    evaluation_data = json.loads(clustering_result.evaluation_results)
                else:
                    evaluation_data = {}
                    
                sil_score = evaluation_data.get('Silhouette Score', None)
                db_score = evaluation_data.get('Davies Bouldin Score', None)
                
                # For Silhouette Score, higher is better
                if sil_score is not None and (best_sil is None or sil_score > best_sil):
                    best_sil = sil_score
                    
                # For Davies Bouldin Score, lower is better
                if db_score is not None and (best_db is None or db_score < best_db):
                    best_db = db_score
            
            # Set the best values or N/A if no valid results
            data[6][embedding_method_name] = best_sil if best_sil is not None else "N/A"
            data[7][embedding_method_name] = best_db if best_db is not None else "N/A"

    
    # Update cache
    _evaluation_cache = data
    _cache_timestamp = current_time
    
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
    if triggered_id == 'new-pipeline':
        return not is_open
    elif triggered_id == 'new-confirm' or triggered_id == 'new-cancel':
        return False  # Close the modal when confirm or cancel is clicked
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
    if triggered_id == 'load-pipeline':
        return not is_open
    elif triggered_id in ['load-cancel', 'load-confirm']:
        return False  # Close modal when cancel or confirm is clicked
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
    [Input("url", "pathname"),
     Input("new-confirm", "n_clicks"),
     Input("load-confirm", "n_clicks")],
    [State("new-methods-embedding", "value"),
     State("new-methods-clustering", "value"),
     State("new-methods-dimred-viz", "value")],
    prevent_initial_call=False
)
def fetch_evaluation_data(pathname, n_clicks, load_clicks, m_e, m_c, m_dv):
    """Fetch evaluation data"""
    from dash import callback_context
    
    triggered = callback_context.triggered_id if callback_context.triggered else None
    
    # Handle new pipeline creation
    if n_clicks and n_clicks > 0 and triggered == "new-confirm":
        selected_dataset = sqlalchemy_db.session.query(Dataset).filter_by(is_selected=True).first()
        if not selected_dataset:
            return extract_evaluation_results()
        
        dataset_name = selected_dataset.dataset_name
        
        if m_e:
            clear_evaluation_cache()
            embedding_method = m_e[0] if isinstance(m_e, list) else m_e
            embedding_instance = get_embedding_model(embedding_method, dataset_name)
            embedding_instance.process()
            evaluation_instance = EmbeddingsEvaluation(embedding_method, None)
            evaluation_instance.evaluate()
            data = extract_evaluation_results()
            if m_c:
                clustering_method = m_c[0] if isinstance(m_c, list) else m_c
                clustering_instance = get_clustering_model(clustering_method, dataset_name=dataset_name, embedding_method=embedding_method)
                clustering_instance.embeddings = clustering_instance.load_data()
                clustering_instance.fit_predict()
                clustering_instance.save_to_database(clustering_method)
                evaluation_instance = ClusteringEvaluation(embedding_method, clustering_method)
                evaluation_instance.evaluate()
                data = extract_evaluation_results()
            if m_dv:
                dim_reduction_method = m_dv[0] if isinstance(m_dv, list) else m_dv
                dim_reduction_instance = get_dr_model(dim_reduction_method)
                dim_reduction_instance.fit_transform(embedding_method)
            return data
        return extract_evaluation_results()
    
    # Handle load pipeline
    if load_clicks and load_clicks > 0 and triggered == "load-confirm":
        return extract_evaluation_results()
    
    # Handle page load to /explore
    if pathname == '/explore' and (triggered == "url" or not triggered):
        clear_evaluation_cache()
        data = extract_evaluation_results()
        return data
    
    # Default case - return empty data
    return []


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
        try:
            embedding_method = m_e[0] if isinstance(m_e, list) else m_e
            clustering_method = m_c[0] if isinstance(m_c, list) else m_c
            dim_reduction_method = m_dv[0] if isinstance(m_dv, list) else m_dv

            figures = {}
            
            # Try to create each visualization with error handling
            try:
                figures["cluster-time-histogram"] = ClusterTemporalHist(embedding_method, clustering_method, dim_reduction_method).plot()
            except Exception as e:
                figures["cluster-time-histogram"] = None

            try:
                figures["cluster-time-grid"] = ClusterTimeGrid(embedding_method, clustering_method, dim_reduction_method).plot()
            except Exception as e:
                figures["cluster-time-grid"] = None

            try:
                figures["time-series"] = TimeSeries(embedding_method, clustering_method, dim_reduction_method).plot()
            except Exception as e:
                figures["time-series"] = None

            try:
                figures["temp-rose-plot"] = RosePlot(embedding_method, clustering_method, dim_reduction_method).plot()
            except Exception as e:
                figures["temp-rose-plot"] = None

            try:
                figures["cluster-state-space"] = StateSpaceVis(embedding_method, clustering_method, dim_reduction_method).plot()
            except Exception as e:
                figures["cluster-state-space"] = None

            return figures, "Figures Fetched. Please view them in the respective Tabs"
            
        except Exception as e:
            return {}, f"Error creating visualizations: {str(e)}"
    
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
    if figure_data and figure_data is not None:
        figure_component = dcc.Graph(figure=figure_data)
    else:
        figure_component = html.Div(f"Figure not available for '{active_tab}'. This visualization may have failed to load.")

    return figure_component, ""


@callback(
    Output('evaluation-table-container', 'children'),
    Input('evaluation-table', 'data'),
    prevent_initial_call=False
)
def update_evaluation_table_display(table_data):
    """Display the evaluation table"""
    
    # Show the actual table
    columns = [
        {"name": "Metric", "id": "metric"},
        {"name": "BirdNET", "id": "birdnet"},
        {"name": "Acoustic Indices", "id": "acoustic_indices"},
        {"name": "VAE", "id": "vae"}
    ]
    
    return dash_table.DataTable(
        id='evaluation-table',
        columns=columns,
        data=table_data if table_data else [],
        merge_duplicate_headers=True,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'}
    )







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
