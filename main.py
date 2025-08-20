from rag import Loader, Vectorizer

if __name__ == "__main__":

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------
    loader = Loader(data_dir="data", exclude_file_types=["csv"])
    data = loader.load_files()
    loader.peek_data(data)

    print(f"Loaded {len(data)} files")

    # ------------------------------------------------------------
    # VECTORIZE DATA
    # ------------------------------------------------------------
    file_name = "history_of_cricket.md"
    
    vectorizer = Vectorizer(data=data, chunk_size=500, chunk_overlap=150)
    vectorizer.vectorize_docs()
    print(f"Number of chunks: {len(vectorizer.chunks[file_name])}")
    print(f"Number of embeddings: {len(vectorizer.embeddings[file_name].data)}")

