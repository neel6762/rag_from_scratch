from rag import Loader, Indexer

if __name__ == "__main__":

    # Load Data
    loader = Loader(data_dir="data", exclude_file_types=["csv"])
    data = loader.load_files()
    loader.peek_data(data)

    print(f"Loaded {len(data)} files")
