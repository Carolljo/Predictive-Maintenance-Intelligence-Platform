import os
import joblib


def create_directory(directory_path):
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    directory_path : str
        Path of the directory to create.

    Returns
    -------
    None
        The directory is created on disk if it does not already exist.
    """
    os.makedirs(directory_path, exist_ok=True)
    
    
    
def save_object(obj, file_path):
    """
    Save a Python object to disk using joblib.

    Parameters
    ----------
    obj
        Python object to save.

    file_path : str
        Path where the object will be saved.

    Returns
    -------
    None
        The object is serialized and saved to disk.
    """
    create_directory(os.path.dirname(file_path))
    joblib.dump(obj, file_path)


def load_object(file_path):
    """
    Load a Python object from disk using joblib.

    Parameters
    ----------
    file_path : str
        Path to the saved object.

    Returns
    -------
    object
        Python object loaded from the specified file.
    """
    return joblib.load(file_path)