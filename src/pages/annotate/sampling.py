import glob
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from flask_login import current_user

from src.utils import glob_audio_dataset
from .model import get_model, train_model

# from dash_app import app_utils

logger = logging.getLogger(__name__)


def get_sample(project_name):
    try:
        path_queue_csv = os.path.join('projects', project_name, 'queue.csv')
        queue = pd.read_csv(path_queue_csv, index_col=0)
        sample = queue.loc[queue['status'] == 'pending', :].sample(1).squeeze()
        queue.loc[queue.loc[:, 'status'] == 'active', 'status'] = 'zombie'
        queue.loc[sample.name, 'status'] = 'active'
        queue.to_csv(path_queue_csv)

        return sample['sound_clip_url']

    except Exception as e:
        logger.exception(e)


def process_annotation(project_name, current_sample, new_queue_status, labels, callback_trigger):
    logger.debug('Not implemented')

    try:
        path_queue_csv = os.path.join('projects', project_name, 'queue.csv')
        queue = pd.read_csv(path_queue_csv, index_col=0)
        logger.debug(
            f"Current sample {current_sample} status active?: {queue.loc[queue['sound_clip_url'] == current_sample, 'status'] == 'active'}")
        queue.loc[queue['sound_clip_url'] == current_sample, 'status'] = new_queue_status
        queue.to_csv(path_queue_csv)

        labels = labels or ['']
        path_anont_csv = os.path.join('projects', project_name, 'annotations.csv')
        annotations = pd.read_csv(path_anont_csv, index_col=0)
        index = range(len(labels)) if annotations.index.empty else range(annotations.index.max() + 1,
                                                                         annotations.index.max() + 1 + len(labels))
        new_annotations = pd.DataFrame({
            'sound_clip_url': current_sample,
            'label': labels,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'username': current_user.username
        }, index=pd.Index(index, name='id'))
        annotations = new_annotations if annotations.empty else pd.concat([annotations, new_annotations])
        annotations.to_csv(path_anont_csv)
    except Exception as e:
        logger.exception(e)


def replenish_queue(n_min, project_name, method, n_max=None):
    path_queue_csv = os.path.join('projects', project_name, 'queue.csv')
    queue = pd.read_csv(path_queue_csv, index_col=0)

    ndx_pending = queue['status'] == 'pending'
    if ndx_pending.sum() < n_min:
        all_clips = glob_audio_dataset(os.path.join('projects', project_name, 'clips'))
        candidate_clips = list(set([os.path.basename(p) for p in all_clips]) - set(queue['sound_clip_url']))

        n_max = n_max or n_min * 2
        n_max = min(n_max, len(candidate_clips))

        if method == 'refine':
            new_samples = _refine(project_name, candidate_clips, n_max)
        elif method == 'explore':
            new_samples = _explore(project_name, candidate_clips, n_max)
        elif method == 'random':
            # priority = 0
            new_samples = np.random.choice(candidate_clips, n_max, replace=False)

        new_to_queue = pd.DataFrame({
            'sound_clip_url': new_samples,
            'status': 'pending',
            'method': method,
            # 'priority': priority,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, index=pd.Index(range(queue.index.max() + 1, queue.index.max() + 1 + n_max), name='id'))

        queue = pd.concat((queue, new_to_queue))
        queue.to_csv(path_queue_csv)

    return (queue['status'] == 'processed').sum()


def _explore(project_name, candidate_clips, n):
    from sklearn.metrics.pairwise import pairwise_distances

    path_embeddings = os.path.join('projects', project_name, 'embeddings.pkl')
    embeddings = pd.read_pickle(path_embeddings)
    path_queue_csv = os.path.join('projects', project_name, 'queue.csv')
    queue = pd.read_csv(path_queue_csv, index_col=0)
    queued_clips = queue['sound_clip_url'].to_list()

    path_distance_matrix = os.path.join('projects', project_name, 'distance_matrix.pkl')
    if os.path.exists(path_distance_matrix):
        distance_matrix = pd.read_pickle(path_distance_matrix)
        # TODO check for missing/superfluous rows and columns, and fix it
    else:
        distance_matrix = pd.DataFrame(
            data=pairwise_distances(embeddings.loc[candidate_clips], embeddings.loc[queued_clips]),
            index=candidate_clips, columns=queued_clips
        )

    new_samples = []
    while len(new_samples) < n:
        next_sample = distance_matrix.min(axis=1).idxmax()
        distance_matrix.drop(next_sample, axis=0, inplace=True)
        new_distances = pd.DataFrame(
            data=pairwise_distances(embeddings.loc[distance_matrix.index], embeddings.loc[next_sample].to_frame().T),
            index=distance_matrix.index, columns=[next_sample]
        )
        distance_matrix = distance_matrix.join(new_distances)
        new_samples.append(next_sample)

    distance_matrix.to_pickle(path_distance_matrix)

    return new_samples


def _refine(project_name, candidate_clips, n):
    path_anont_csv = os.path.join('projects', project_name, 'annotations.csv')
    annotations = pd.read_csv(path_anont_csv, index_col=0)

    path_embeddings = os.path.join('projects', project_name, 'embeddings.pkl')
    embeddings = pd.read_pickle(path_embeddings)
    annotations['value'] = 1  # (~annotations['label'].isnull()).astype(int)
    y_train = annotations.fillna('NaN').pivot_table(index='sound_clip_url', values='value', columns='label',
                                                    fill_value=0).astype(bool).drop('NaN', axis=1)
    x_train = embeddings.loc[y_train.index]
    x_sample = embeddings.loc[candidate_clips]

    mdl = get_model(num_outputs=len(y_train.columns), input_shape=x_train.iloc[0, :].shape)
    train_model(mdl, x_train, y_train)
    y_pred_sample = mdl.predict(x_sample)

    uncert_ratio_max = pd.Series(
        data=np.max(1 / (0.5 + np.abs(y_pred_sample - 0.5)) - 1, axis=1),
        index=x_sample.index,
        name='uncert_ratio_max'
    )
    new_samples = uncert_ratio_max.sort_values(ascending=False).iloc[:n].index.to_list()

    return new_samples

# def _validation(annotations, detections, competence_classes=None, sampling_selected_species=None,
#                 manual_col_prefix='species_', col_processed='processed', col_skipped='skipped'):
#     # get all species columns
#     species_columns = [col for col in annotations.columns if col.startswith(manual_col_prefix) and
#                        any(col.startswith(comp_class) for comp_class in competence_classes)]
#     species_columns = [s.replace(manual_col_prefix, '', 1) for s in species_columns]
#
#     # drop species that are detected 5 or more times
#     species_to_drop = []
#     # TODO Vectorize
#     for col in species_columns:
#         if annotations[manual_col_prefix + col].sum() >= 5:
#             species_to_drop.append(col)
#     species_columns = list(set(species_columns) - set(species_to_drop))
#
#     # drop species that are manually excluded
#     species_columns = list(set(species_columns) - set(sampling_selected_species))
#
#     # no species column in competence class
#     if not species_columns:
#         logger.info("No categories found for validation. Returning random sample.")
#         # TODO Define indices_to_sample within scope of validation()
#         raise NotImplementedError
#         return random(indices_to_sample)
#
#     # get df with relevant columns
#     columns_to_keep = [col for col in detections.columns if any(sub in col for sub in species_columns)]
#     relevant_detections = detections[columns_to_keep].copy()
#
#     # get def with relevant rows
#     unprocessed_mask = (annotations[col_processed] != 1) & (annotations[col_skipped] != 1)
#     relevant_detections = relevant_detections[unprocessed_mask]
#
#     # get index of highest score
#     max_value = relevant_detections.values.max(axis=1)
#     row_index = np.random.choice(len(max_value), p=max_value / max_value.sum())
#     return row_index
#
#

#
#
# def _discover(annotations, embeddings, competence_classes=None, sampling_selected_species=None,
#               manual_col_prefix='species_', col_processed='processed', col_skipped='skipped'):
#     # divide embeddings in labelled and unlabelled samples
#     indices_labelled = annotations.index[annotations[col_processed] == 1].tolist()
#     indices_unlabelled = annotations.index[annotations[col_processed] == 0].tolist()
#
#     # exclude all species columns not optimised for
#     training_columns = [col for col in annotations.columns if col.startswith(manual_col_prefix) and
#                         any(comp_class in col for comp_class in competence_classes)]
#
#     # # if less than 10 samples are labelled, choose other sampling method
#     # if len(indices_labelled) < 10:
#     #     return _sampling_validation()
#     # elif not training_columns:
#     #     return _sampling_random()
#
#     # get metadata
#     training_df = annotations.loc[indices_labelled, training_columns]
#     y_train = training_df.to_numpy()
#     # get training and sampling data
#     x_train = embeddings[indices_labelled, :]
#     x_sample = embeddings[indices_unlabelled, :]
#
#     # detection model: get y labels
#     y_sampled_species_present = y_train.any(1)
#     # detection model: create model
#     model_detection = create_model_mil(shape=x_train.shape[1:], units=1)
#     # detection model: train model
#     train_model(model_detection, x_train, y_sampled_species_present)
#     # detection model: get predictions
#     y_pred_detection = model_detection(x_sample)
#     y_pred_detection = np.max(y_pred_detection, axis=1)
#
#     # identification model: get y labels
#     indices_present_classes = np.nonzero(np.sum(y_train, axis=0))[0]
#     y_sampled_nonempty_classes = y_train[:, indices_present_classes]
#     # identification model: create model
#     model_classification = create_model_mil(shape=x_train.shape[1:], units=len(indices_present_classes))
#     # identification model: train model
#     train_model(model_classification, x_train, y_sampled_nonempty_classes)
#     # identification model: get predictions
#     y_pred_classification = model_classification(x_sample)
#     y_pred_classification_max = np.max(y_pred_classification, axis=1)
#
#     # score combination (most certain a detection + most certain no classification)
#     sample_score = y_pred_detection * (1 - y_pred_classification_max)
#     logit_sample_score = np.log(sample_score / (1 - sample_score))
#     logit_sample_score -= logit_sample_score.min()
#     logit_sample_score /= logit_sample_score.sum()
#
#     # softmax selection
#     sampled_embedding_index = np.random.choice(len(sample_score), p=logit_sample_score)
#     sampled_index = indices_unlabelled[sampled_embedding_index]
#     return sampled_index
