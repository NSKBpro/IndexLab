# IndexLab

[![Build Status](https://github.com/NSKBpro/IndexLab/actions/workflows/ci.yml/badge.svg)](https://github.com/NSKBpro/IndexLab/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**IndexLab** is a flexible, open-source toolkit for **building, searching, and evaluating vector indexes**. It’s designed for engineers and researchers who want to experiment with embedding retrieval, index performance, and similarity search at scale.

> ⚠️ **Work in Progress**: IndexLab is actively developed. APIs and features may change. Several major capabilities (GPU acceleration, new metrics, plugin system) are on the roadmap.

---

## 🚀 Table of Contents

1. [Motivation](#motivation)  
2. [Key Features](#key-features)  
3. [Architecture](#architecture)  
4. [Getting Started](#getting-started)  
5. [Usage Examples](#usage-examples)  
6. [Configuration](#configuration)  
7. [API / CLI](#api--cli)  
8. [Testing](#testing)  
9. [Roadmap](#roadmap)  
10. [Contributing](#contributing)  
11. [License](#license)  
12. [Acknowledgments](#acknowledgments)  
13. [Contact](#contact)

---

## Motivation

Modern AI pipelines depend heavily on vector search (semantic retrieval, recommendations, RAG, etc.). Experimenting with multiple index strategies and benchmarking them is time-consuming and fragmented. **IndexLab** streamlines this process:

- Quick prototyping of new index configurations  
- Side-by-side performance & accuracy comparisons  
- Modular components (indexers, retrievers, evaluators)  
- Extensible metrics and logging  

---

## Key Features

- 🔍 Build, persist, and query vector indexes with pluggable backends  
- 📊 Benchmarking & evaluation metrics (latency, recall, memory usage)  
- 🧪 Support for multiple vector indexing algorithms (HNSW, IVF, PQ, etc.)  
- 🔄 Hot-swappable modules (distance metrics, quantizers)  
- 🐳 Docker / Compose support for easy deployment  
- 📂 Data / embedding ingestion utilities  
- 📈 Visualization & metrics export for comparisons  

---

## Architecture

