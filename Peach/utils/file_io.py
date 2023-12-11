# Package the iopath library to make it easier to use
from iopath.common.file_io import PathManager as PathManagerBase
from iopath.common.file_io import HTTPURLHandler, OneDrivePathHandler, PathHandler
import pickle

__all__ = ["PathManager", "PathHandler"]
PathManager = PathManagerBase()
PathManager.register_handler(HTTPURLHandler())
PathManager.register_handler(OneDrivePathHandler())



def load_pickle_from_path(path):
    """Load a pickle file from a given path.

    Args:
        path (str): Path to the pickle file.

    Returns:
        Any: The loaded pickle object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)