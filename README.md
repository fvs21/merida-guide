---
title: Merida Guide
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
---

Aim: An AI-powered tourist assistant for Mérida, Yucatán using Retrieval Augmented Generation and local data.

# Architecture / Tech Stack

* Frontend: Gradio
* Backend: HuggingFace embeddings, HuggingFace Inference API, ChromaDB for vector database and LangChain

# Overview

Mérida Guide is an intelligent tourist assistant for Mérida, Yucatán, powered by a Retrieval-Augmented Generation (RAG) pipeline. It answers questions about food, transport, entertainment, and locations in the city using context-rich local data and Google Maps integration.