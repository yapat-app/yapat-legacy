# Frontend

The YAPAT frontend is a React application that provides a timeline-based
annotation interface on top of the backend API.

---

## User Stories

Three users illustrate how YAPAT works in practice: **setup**, **annotation**, and **refinement** — all centered around the same workspace, the **Feed**.

### User A — The Setup

User A is the dataset owner.
They create a **team** and invite User B to join.
Next, they register a **dataset** whose `.wav` files have already been copied to disk.
The dataset comes from a passive acoustic monitoring (PAM) deployment with sixteen recorders in the same biosphere reserve core area.

Ideally, each recording would include metadata about the recorder device and its environment, but that can wait.
During setup, the recordings are **split into snippets** using default parameters and **embedded** with the default model.
User A may also predefine a few **GBIF taxa of interest** to guide annotation.
That’s all that is required from User A.

### User B — The Pioneer Annotator

User B is an expert listener. They can recognize calls of the taxa of interest, either by ear or from their spectrograms. They also hold several **reference recordings** from a trusted sound library.

After logging in, User B joins the team created by User A, selects the dataset, and begins annotating within the **Feed** — the main workspace of YAPAT.

They start with an **exploratory feed**, using the *random* mode to draw twenty snippets from the dataset.
Out of these, only three contain recognizable species; those receive GBIF-linked annotations, while the rest are left unlabelled.

Next, User B creates a **similarity-based feed** to find more examples of a species they just annotated.
They select the *similarity* method, which opens an upload dialog.
After uploading a **focal recording**, the tool displays a **histogram** of the file with a **fixed-size selection box** (defined by the snippet configuration).
User B positions the box over the most representative part of the call.
The system extracts that snippet, computes embeddings, and builds a feed with the *n* most similar snippets (default: cosine similarity, with an option to switch to Euclidean).

User B repeats this for several species and then **pushes their annotations** to the team database before ending the session.

### User C — The Model Refiner

User C logs in later and sees that new annotations are available for the same dataset.
They **pull the updated annotations** and open the **Feed** again — this time choosing a mode that emphasizes **active refinement**.

Without specifying particular taxa, the system assembles a mixed feed containing samples from all species annotated by User B, prioritizing uncertain or informative snippets.
User C performs a **multi-label active learning session**, verifying or correcting predictions directly within the feed.
When finished, they also **push annotations** back to the shared database.

---

The story highlights a simple principle:

> **Everything happens through the Feed.**
> Whether exploring, searching by similarity, or refining models, the expert remains in control — the machine only helps decide what to look at next.

---


## Overview

- Stack: React, TypeScript (if applicable), fetch/axios for HTTP.
- Core concepts: feeds, snippet cards, local-first annotation.

## React Components

### AnnotationFeed



### SnippetCard



## State and API Integration

Brief description of how global state is managed (e.g. React Query, Redux, Zustand, etc.),
how authentication tokens are stored, and how requests to the backend API are made.

Optionally add a small code snippet:

```ts
// Example: fetching snippets for a feed

const response = await fetch("/api/snippets?feed_id=...");
const data = await response.json();
