# Frontend

The YAPAT frontend is a React application that provides a timeline-based
annotation interface on top of the backend API.

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
