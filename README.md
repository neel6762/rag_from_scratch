# RAG from Scratch

This project is a simple implementation of a Retrieval-Augmented Generation (RAG) system built from scratch in Python.

## Overview

The system is designed to ingest documents from a local directory, process them, and prepare them for a retrieval and generation pipeline. It currently supports using different Large Language Models (LLMs) like OpenAI's GPT models and local models via Ollama.

## Features

*   **Document Loading**: Loads various file types (e.g., PDFs, text files) from a specified data directory.
*   **Configurable LLMs**: Easily switch between LLM providers (currently supports OpenAI and Ollama).

## Setup

### Prerequisites

*   Python 3.12+
*   An LLM provider:
    *   For **OpenAI**: An API key.
    *   For **Ollama**: A running Ollama instance.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/rag_from_scratch.git
    cd rag_from_scratch
    ```

2.  **Install dependencies using [uv](https://github.com/astral-sh/uv):**
    This project uses `uv` for package management. To create a virtual environment and install dependencies, run the following commands:
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    uv pip install -e .
    ```

### Configuration

1.  **Create a `.env` file** in the root of the project.

2.  If you are using **OpenAI**, add your API key to the `.env` file:
    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```
    If you are using **Ollama**, no environment variable is needed if it's running on the default `http://localhost:11434`.

## Usage

1.  **Add your documents:**
    Create a `data` directory in the project root and place the files you want to process inside it.

2.  **Run the application:**
    ```bash
    uv run python main.py
    ```
    This will load the documents from the `data` directory.

## Current Status

This project is under active development. The current focus is on building out the core functionalities.
