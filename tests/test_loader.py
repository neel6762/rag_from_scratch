from rag import Loader


def test_loader_load_files():

    loader = Loader(data_dir="data")
    data = loader.load_files()

    assert data is not None, "Data should not be None"
    assert isinstance(data, dict), "Data should be a dictionary"

    assert len(data) > 0, "Data should have at least one file"


def test_loader_load_files_with_exclude_file_types():

    loader = Loader(data_dir="data", exclude_file_types=["pdf"])
    data = loader.load_files()

    for key in data.keys():
        assert key.split(".")[-1] != "pdf", f"{key} should not be in data (should not include pdf files)"


def test_loader_load_files_with_exclude_file_names():

    loader = Loader(data_dir="data", exclude_file_names=["history_of_cricker.md"])
    data = loader.load_files()

    assert "history_of_cricker.md" not in data.keys(), "history_of_cricker.md should not be in data"

if __name__ == "__main__":
    test_loader_load_files()
    test_loader_load_files_with_exclude_file_types()
    test_loader_load_files_with_exclude_file_names()