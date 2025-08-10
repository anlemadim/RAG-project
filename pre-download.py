"""Helper script to pre-download Sentence Transformer models.

This file exists so the SentenceTransformer model used in the labs can be
downloaded ahead of time. Previously the import was executed at the module
level, which caused a ``ModuleNotFoundError`` in environments where
``sentence_transformers`` hadn't been installed yet.  By moving the import
inside a function we avoid that failure and provide a clearer error message.
"""


def download_model() -> None:
    """Download the embedding model if the dependency is available."""
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise SystemExit(
            "sentence-transformers is not installed. Please install the "
            "required dependencies before running this script."
        ) from exc

    SentenceTransformer("all-MiniLM-L6-v2")  # Model to create embeddings


if __name__ == "__main__":  # pragma: no cover - script guard
    download_model()
