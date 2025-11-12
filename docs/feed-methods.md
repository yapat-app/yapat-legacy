# Feed Methods

This page details the supported **sampling methods** available in YAPAT’s `Feed API`.
All methods operate through the same endpoint:

```text
GET /datasets/{dataset_id}/feed
```

and share a common response format, as described in [Backend API → Feed Generation](backend-api.md#feed-generation).

---

## `random`

### Description

Randomly samples snippets from the dataset without bias.
Useful for initial exploration or broad manual labeling before model initialization.

### Parameters

| Parameter | Type   | Description                                              |
| --------- | ------ | -------------------------------------------------------- |
| `limit`   | int    | Maximum number of snippets to return (default 50).       |
| `status`  | string | Filter by snippet status (`unlabeled`, `labeled`, etc.). |

### Notes

* Provides an unbiased baseline for annotation.
* Often used at the very beginning of a project to seed labeled data.
* High implementation priority.

---

## `explore`

### Description

Selects snippets to **maximize diversity** in the embedding space, ensuring coverage across different acoustic conditions or species.
Implemented via clustering or core-set sampling.

### Parameters

| Parameter            | Type   | Description                                                        |
| -------------------- | ------ | ------------------------------------------------------------------ |
| `limit`              | int    | Maximum number of snippets to return.                              |
| `embedding_model_id` | string | Optional override (defaults to dataset’s current embedding model). |

### Notes

* Intended to “cover the space” early on.
* Works best once embeddings are available for all snippets.
* Low implementation priority.

---

## `history`

### Description

Retrieves snippets that already have annotations, filtered and sorted for review or export.

### Parameters

| Parameter      | Type     | Description                                  |
| -------------- | -------- | -------------------------------------------- |
| `species[]`    | list     | GBIF taxon keys to filter by.                |
| `annotators[]` | list     | User IDs to filter by.                       |
| `from`, `to`   | ISO 8601 | Optional time range for annotation creation. |
| `sort_by`      | string   | Sort field (e.g. `created_at`).              |
| `sort_dir`     | string   | Sort direction (`asc` or `desc`).            |
| `limit`        | int      | Max items to return.                         |

### Notes

* Used for reviewing or exporting annotated data.
* Complements the `/datasets/{id}/annotations` endpoint.
* Medium implementation priority.

---

## `initialize_model`

### Description

Bootstraps a **classifier manifest** for one or more target taxa using provided positives (and optional negatives).
The backend reuses existing per-taxon heads where available and **asynchronously trains** any missing heads, then returns candidate snippets to review after initialization.

### Parameters

| Parameter                | Type   | Description                                                              |
| ------------------------ | ------ | ------------------------------------------------------------------------ |
| `target_taxa[]`          | list   | **Namespaced taxon IDs** (e.g., `gbif:2420576`) to model.                |
| `positive_example_ids[]` | list   | IDs of `TrainingExample`s representing positives for the requested taxa. |
| `negative_example_ids[]` | list   | Optional IDs of examples to serve as background/negatives.               |
| `embedding_model_id`     | string | Optional override (defaults to dataset’s current embedding model).       |
| `limit`                  | int    | Maximum number of candidate snippets to return.                          |

### Notes

* Produces/updates a **classifier manifest** internally (equivalent to `POST /classifiers`) and then surfaces high-value candidates to label.
* Uses the dataset’s active **`SnippetConfig`** and the specified (or default) **`EmbeddingModel`**.
* Implementation priority is lower than `similarity`. 

---

## `validate`

### Description

Retrieves snippets for which an existing classifier is **most confident** in its predictions so users can confirm or correct labels.
The classifier must be `ready` and tied to the same dataset/representation as the feed.

### Parameters

| Parameter         | Type   | Description                                                                 |
| ----------------- | ------ | --------------------------------------------------------------------------- |
| `classifier_id`   | string | **Required** — the classifier (manifest) to validate.                       |
| `target_taxon_id` | string | **Optional** — namespaced taxon ID to focus on (must be in classifier set). |
| `limit`           | int    | Number of top-confidence snippets to return.                                |

**Constraints**

* `target_taxon_id`, if provided, **must** be present in `classifier.target_taxa`.
* If omitted, per-snippet **top head** among `classifier.target_taxa` is used for ranking.

### Notes

* High-confidence predictions help estimate precision and reveal systematic errors.
* Feed items may include: `"scores": { "classifier_confidence": 0.98 }`.
* Often paired with `refine` for active learning. 

---

## `refine`

### Description

Selects snippets where the classifier is **least certain**, for efficient active learning.
The classifier must be `ready` and tied to the same dataset/representation as the feed.

### Parameters

| Parameter         | Type   | Description                                                                 |
| ----------------- | ------ | --------------------------------------------------------------------------- |
| `classifier_id`   | string | **Required** — the classifier (manifest) to refine.                         |
| `target_taxon_id` | string | **Optional** — namespaced taxon ID to focus on (must be in classifier set). |
| `limit`           | int    | Maximum number of snippets to return.                                       |

**Constraints**

* `target_taxon_id`, if provided, must be present in `classifier.target_taxa`.
* If omitted, uncertainty is computed over the classifier’s **multilabel heads**, e.g., `1 - max_t p(t)` or margin.

### Notes

* Typically implemented by sorting by an uncertainty score (e.g., `1 - p_max`, logit margin).
* Feed items may include: `"scores": { "classifier_confidence": 0.55, "classifier_uncertainty": 0.45 }`.
* Works best after some positives/background exist. 

---

## `similarity`

### Description

Finds snippets acoustically similar to a short **uploaded query example**.
The example is embedded on the fly using the dataset’s active **`SnippetConfig`** and **`EmbeddingModel`**; the example is ephemeral and not stored.

### Parameters (multipart or JSON)

| Parameter        | Type          | Description                                                |
| ---------------- | ------------- | ---------------------------------------------------------- |
| `query_file`     | file / base64 | Uploaded query audio (short, e.g., 3 s).                   |
| `crop_start_sec` | float         | Optional crop start time.                                  |
| `crop_end_sec`   | float         | Optional crop end time.                                    |
| `limit`          | int           | Maximum number of similar snippets to return (default 50). |

### Notes

* Uses cosine (or equivalent) similarity in the embedding space.
* Stateless — no DB records are created; vectors remain internal.
* Corresponds to the **“Find similar”** mode in the UI. 


---

## `discover`

### Description

Identifies **biophonic snippets** that are not well explained by existing classifiers.
This helps surface potential new taxa or acoustic events.

### Parameters

| Parameter            | Type  | Description                                        |
| -------------------- | ----- | -------------------------------------------------- |
| `min_biophony_score` | float | Optional — lower bound for biophonic confidence.   |
| `max_explained_prob` | float | Optional — upper bound for known-class confidence. |
| `limit`              | int   | Maximum number of snippets to return.              |

### Notes

* Aims to find “unknown unknowns” — interesting new acoustic events.
* See Hannes's paper at GoodIT 2023.
* Low implementation priority.