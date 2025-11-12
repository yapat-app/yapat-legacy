# YAPAT — Yet Another PAM Annotation Tool

YAPAT is a lightweight, human-in-the-loop system for the expert annotation of
**Passive Acoustic Monitoring (PAM)** data.  
It provides an efficient workflow for exploring, labeling, and managing large
audio datasets used in biodiversity monitoring and eco-acoustic research.

---

## Motivation

Passive Acoustic Monitoring has become an essential method for assessing
animal biodiversity, allowing continuous, minimally invasive observation of
ecosystems at scale.  
While modern machine learning techniques are increasingly applied to PAM data,
annotation tools have not kept pace.  
YAPAT fills this gap by integrating human expertise with interactive sampling
and model-assisted labeling.

---

## Architecture Overview

YAPAT combines a **FastAPI** backend with a **React.js** frontend connected via
a lightweight REST interface.

- **Backend**: handles data ingestion, snippet generation, annotation storage,
  and ontology integration (GBIF for species, future support for ENVO and IoT).
- **Frontend**: provides a timeline-style annotation interface for reviewing
  short audio snippets, designed for speed and offline resilience.
- **Synchronization**: follows a *local-first* model—annotations are stored
  locally and synchronized explicitly through `fetch`, `pull`, and `push`
  operations.
- **Extensibility**: modular design allows new ontologies, models, and
  annotation modes to be integrated over time.

---

## Documentation Structure

This developer documentation is organized into two main sections:

| Section | Description |
|:---------|:-------------|
| **Backend API** | Reference for the FastAPI endpoints, data models, and ontology interfaces. |
| **Frontend** | Overview of the React components, application state, and API integration patterns. |

For more detailed architectural context, see the internal development guide.
