import os
import joblib


def create_directory(directory_path):
    """
    Create a directory if it does not already exist.

    Parameters:
        directory_path (str): Path of the directory to create.
    """
    os.makedirs(directory_path, exist_ok=True)
    
    
    
def save_object(obj, file_path):
    """
    Save a Python object to disk using joblib.

    Parameters:
        obj: The Python object to save.
        file_path (str): The location where the object will be saved.
    """
    create_directory(os.path.dirname(file_path))
    joblib.dump(obj, file_path)


def load_object(file_path):
    """
    Load a Python object from disk using joblib.

    Parameters:
        file_path (str): Location of the saved object.

    Returns:
        The loaded Python object.
    """
    return joblib.load(file_path)