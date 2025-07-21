from rag import Loader, Indexer

if __name__ == "__main__":

    # Load Data
    loader = Loader(data_dir="data")
    data = loader.load_files()
    loader.peek_data(data)

    print(f"Loaded {len(data)} files")

    # Index Data
    # indexer = Indexer(data=data)
    # indexer.index_data()