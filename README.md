# PDF to Embedding Search

This project demonstrates how to extract text from a PDF, create sentence embeddings using `Sentence-Transformers`, and perform similarity searches on those embeddings using `FAISS`.

## Features
- Extract text content from PDF files.
- Generate sentence embeddings for text.
- Store and search embeddings using FAISS for fast similarity search.

## Tech Stack
- PyMuPDF (`pymupdf`) for PDF text extraction.
- `sentence-transformers` for generating sentence embeddings.
- `faiss-cpu` for efficient similarity search.

## Setup
1. Install the required libraries:
    ```
    pip install -r requirements.txt
    ```

2. Use the `extract_text_from_pdf()` function to extract text from a PDF file.

3. Generate embeddings using the `SentenceTransformer` model and store them in a FAISS index for fast similarity search.

4. Perform a search using the `search_similar()` function to find sentences similar to a given query.

## Usage
1. Set the path to your PDF file in the script.
2. Run the script to extract the text, generate embeddings, and perform a similarity search.

> Ensure you have the necessary libraries installed and configured.
