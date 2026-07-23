from sklearn.ensemble import RandomForestClassifier

from src.utils import save_object, load_object


def train_model(X_train, y_train):
    """
    Train a Random Forest classifier.

    Parameters:
        X_train: Training features.
        y_train: Training target.

    Returns:
        RandomForestClassifier: Trained model.
    """

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def save_model(model, file_path):
    """
    Save the trained model.

    Parameters:
        model: Trained machine learning model.
        file_path (str): Path to save the model.
    """

    save_object(model, file_path)


def load_model(file_path):
    """
    Load a trained model.

    Parameters:
        file_path (str): Path to the saved model.

    Returns:
        Trained machine learning model.
    """

    return load_object(file_path)