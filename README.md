# Predictive Maintenance Intelligence Platform

## Project Overview

This project is a machine learning based predictive maintenance system built to identify different types of machine failures from operational and sensor data.

The project uses the AI4I-PMDI dataset, which contains machine information such as air temperature, process temperature, rotational speed, torque, tool wear, machine type, and diagnostic labels. The dataset also contains missing and irregular measurements, which made data cleaning and preprocessing an important part of the project.

The work started with understanding and cleaning the data, followed by exploratory data analysis, feature engineering, preprocessing, model comparison, hyperparameter tuning, class imbalance experiments, and model explainability. After the experimentation stage, the final workflow was converted into reusable Python modules for training and prediction.

The current model is a Random Forest classifier. The project also saves the trained model, preprocessing pipeline, and label encoder so that the same transformations used during training can be reused when making predictions on new machine data.

## Problem Statement

Unexpected machine failures can lead to production downtime, maintenance costs, and loss of productivity. Sensor and operational data can be used to identify patterns associated with different types of machine failures before maintenance decisions are made.

The goal of this project is to build a multiclass machine learning system that uses machine and sensor information to classify the machine's diagnostic condition. The project also focuses on building a reproducible workflow so that the same preprocessing and feature engineering used during model training can be applied to new machine data.

## Project Objectives

- Understand and clean imperfect machine sensor data.
- Explore relationships between operational conditions and machine failures.
- Create useful features from the available sensor and date information.
- Build a reusable preprocessing pipeline for numerical and categorical features.
- Compare machine learning models and evaluate their performance beyond accuracy alone.
- Investigate hyperparameter tuning and class imbalance techniques such as SMOTE.
- Explain the final model using feature importance and SHAP analysis.
- Convert the notebook experiments into reusable Python modules.
- Save the trained model and preprocessing artifacts for inference on new machine data.
- Extend the system with a REST API and AWS deployment.

## Current System Architecture

The current implementation separates experimentation, model training, and inference. Data preparation and model experiments are documented in notebooks, while the reusable training and prediction logic is implemented inside the `src` package.

```text
Raw AI4I-PMDI Dataset
        |
        v
Data Understanding & Cleaning
        |
        v
cleaned_data.csv
        |
        v
Feature Engineering
        |
        +--> Date-based features
        +--> Temperature Difference
        +--> Power Index
        |
        v
Train / Test Split
        |
        v
Preprocessing
        |
        +--> Numerical: Median Imputation + Standard Scaling
        |
        +--> Categorical: Most-Frequent Imputation + One-Hot Encoding
        |
        v
Random Forest Classifier
        |
        v
Model Evaluation
        |
        +--> Accuracy
        +--> Macro F1
        +--> Weighted F1
        +--> Classification Report
        +--> Confusion Matrix
        |
        v
Saved Artifacts
        |
        +--> best_model.pkl
        +--> preprocessor.pkl
        +--> label_encoder.pkl
        |
        v
Prediction Pipeline
        |
        v
Failure Type + Confidence Score
```

## Dataset

The project uses the **AI4I-PMDI** dataset, an enhanced version of the AI4I predictive maintenance dataset. It contains 10,000 machine observations with operational, sensor, machine, and diagnostic information.

The main variables used in the project include:

- **Air temperature (K)** – ambient temperature around the machine.
- **Process temperature (K)** – temperature associated with the machine process.
- **Rotational speed (rpm)** – rotational speed of the machine.
- **Torque (Nm)** – torque generated during operation.
- **Tool wear (min)** – accumulated tool usage/wear.
- **Type** – machine/product quality category.
- **System** and **Control** – operational information included in the dataset.
- **Date** – timestamp associated with the observation.
- **Diagnostic** – target variable representing the machine condition or failure type.

### Target Classes

`Diagnostic` is a multiclass target containing six categories:

- No failure
- Heat Dissipation Failure
- Overstrain Failure
- Power Failure
- Tool Wear Failure
- Random Failures

The dataset is highly imbalanced. After cleaning, the class distribution was:

| Diagnostic Class | Samples |
|---|---:|
| No failure | 9,652 |
| Heat Dissipation Failure | 106 |
| Overstrain Failure | 98 |
| Power Failure | 83 |
| Tool Wear Failure | 42 |
| Random Failures | 19 |

This imbalance is important when evaluating the model. Overall accuracy can appear very high because most observations belong to the `No failure` class, so macro F1, per-class recall, the classification report, and the confusion matrix were also considered during model evaluation.

### Data Quality

The original dataset contains substantial missing values and irregular measurements across several sensor variables. Instead of modifying the original CSV, the raw dataset was preserved in `data/raw/`, and the cleaned dataset was saved separately in `data/processed/`.

This keeps the original data available for reference while allowing the modelling workflow to work from a consistent cleaned dataset.

## Data Preparation

The raw dataset was first inspected for data types, missing values, duplicate records, unique values, and target distribution. Cleaning was performed separately from the original dataset so that the raw data remained unchanged.

The cleaned dataset is stored as `data/processed/cleaned_data.csv` and is used as the starting point for the reusable model training pipeline.

During preprocessing:

- Numerical features are imputed using the **median**.
- Numerical features are scaled using **StandardScaler**.
- Categorical features are imputed using the **most frequent value**.
- Categorical features are encoded using **OneHotEncoder** with unknown-category handling.
- The preprocessing pipeline is fitted only on the training data and then reused to transform the test data.

Fitting preprocessing only on the training set helps prevent information from the test set from leaking into model training.

## Feature Engineering

Additional features were created from the original machine and timestamp information.

### Temperature Difference

`Temperature_Difference` represents the difference between process temperature and air temperature:

`Process temperature (K) - Air temperature (K)`

This provides a direct measure of the temperature gap between the machine process and its surrounding environment.

### Power Index

`Power_Index` is calculated from rotational speed and torque:

`Rotational speed (rpm) × Torque (Nm)`

It is used as an engineered indicator of the machine's operating load. In the final Random Forest feature-importance analysis, Power Index was the highest-ranked feature.

### Date Features

The original `Date` column is converted into:

- Year
- Month
- Day
- Day of Week
- Quarter

The original timestamp is removed after these features are generated.

These date-derived variables are retained in the current model, although their real-world predictive value requires further validation. They may capture patterns specific to the dataset rather than physical machine behaviour, so they should not automatically be interpreted as causal indicators of machine failure.

## Model Development and Experiments

Model development was carried out in stages rather than selecting a single algorithm from the beginning.

### Baseline Model Comparison

Multiple classification models were explored during the modelling stage to establish baseline performance. The comparison considered more than overall accuracy because the target classes are highly imbalanced.

Random Forest was selected as the main baseline model and was later used as the final model in the reusable training pipeline.

### Hyperparameter Tuning

Hyperparameter tuning was performed as a separate experiment to determine whether adjusting the Random Forest configuration could provide a meaningful improvement over the baseline model.

The tuned model did not provide enough improvement to justify replacing the simpler baseline Random Forest. For this reason, tuning was kept as an experiment rather than automatically treating the tuned model as the production model.

### Class Imbalance and SMOTE

Because failure observations are much less common than the `No failure` class, SMOTE was also investigated as an approach to class imbalance.

The purpose of this experiment was to determine whether synthetic oversampling could improve recognition of minority failure classes. The SMOTE experiment did not produce a sufficiently reliable overall improvement to replace the selected baseline model.

This was treated as a model-selection experiment rather than assuming that oversampling would automatically improve the final system.

### Final Model

The final reusable training pipeline uses a **Random Forest classifier** with a fixed `random_state=42`.

The fitted model is stored as:

`artifacts/best_model.pkl`

The preprocessing pipeline and target label encoder are also saved separately so that prediction uses the same transformations and label mapping established during training.

## Model Evaluation

The final Random Forest model was evaluated on a stratified 20% test split containing 2,000 observations.

The production training pipeline produced the following results:

| Metric | Score |
|---|---:|
| Accuracy | 0.9925 |
| Macro F1 | 0.6572 |
| Weighted F1 | 0.9897 |

The high overall accuracy and weighted F1 are strongly influenced by the large number of `No failure` observations in the dataset. Macro F1 is therefore an important metric for this project because it gives equal importance to each diagnostic class regardless of its frequency.

### Per-Class Performance

The final classification report showed strong performance for the majority class and several failure categories. However, the two rarest classes in the test set were not successfully detected.

| Encoded Class | Test Samples | Recall | F1 Score |
|---|---:|---:|---:|
| 0 | 21 | 1.00 | 1.00 |
| 1 | 1,930 | 1.00 | 1.00 |
| 2 | 20 | 0.90 | 0.95 |
| 3 | 17 | 1.00 | 1.00 |
| 4 | 4 | 0.00 | 0.00 |
| 5 | 8 | 0.00 | 0.00 |

The confusion matrix showed that all samples belonging to classes 4 and 5 were classified as the dominant class or another class.

### Interpretation

The model performs well on the majority class and several failure categories, but the 99.25% accuracy should not be interpreted as equal performance across all six diagnostic classes.

The main limitation is the severe class imbalance. Some failure categories contain very few observations, which gives the model limited examples from which to learn their patterns.

For this reason, the project reports macro F1 and per-class performance alongside accuracy rather than using accuracy as the only measure of model quality.

Hyperparameter tuning and SMOTE were investigated during experimentation, but neither provided sufficient evidence to replace the selected baseline Random Forest.

Further improvement would require stronger validation of minority-class performance, potentially using additional failure observations, alternative imbalance-handling methods, cost-sensitive learning, or different modelling approaches.

## Model Explainability

Model explainability was included to understand which features had the strongest influence on the Random Forest model rather than treating the classifier as a complete black box.

Two forms of explainability were explored during the project.

### Random Forest Feature Importance

The reusable `src/explainability.py` module extracts feature names from the fitted preprocessing pipeline and matches them with the Random Forest's built-in feature importance scores.

The highest-ranked features from the final model were:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | Power Index | 0.2175 |
| 2 | Torque (Nm) | 0.1530 |
| 3 | Rotational speed (rpm) | 0.1470 |
| 4 | Tool wear (min) | 0.1278 |
| 5 | Temperature Difference | 0.0854 |
| 6 | System | 0.0482 |

The results suggest that operating load, torque, rotational speed, tool wear, and temperature behaviour are important signals used by the model.

Feature importance describes how useful a feature was to the Random Forest when making splits. It should not be interpreted as proof that a feature directly causes machine failure.

### SHAP Analysis

SHAP was explored separately in the model explainability notebook to provide additional insight into how features contribute to model predictions.

This complements the Random Forest feature-importance analysis by examining feature contributions rather than relying only on the model's built-in importance scores.

The explainability work is documented in:

`notebooks/09_Model_Explainability.ipynb`

## Project Structure

The project separates exploratory notebook work from reusable machine learning code. Notebooks document the experimentation process, while the `src` package contains the modular training, evaluation, explainability, and prediction workflow.

```text
Predictive-Maintenance-Intelligence-Platform/
│
├── artifacts/
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   ├── preprocessor.pkl
│   └── tuned_random_forest.pkl
│
├── data/
│   ├── raw/
│   │   └── AI4I-PMDI.csv
│   │
│   └── processed/
│       ├── cleaned_data.csv
│       └── engineered_data.csv
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_cleaning.ipynb
│   ├── 03_EDA.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Preprocessing.ipynb
│   ├── 06_Model_Building.ipynb
│   ├── 07_Hyperparameter_Tuning.ipynb
│   ├── 08_SMOTE.ipynb
│   └── 09_Model_Explainability.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
│   ├── explainability.py
│   ├── pipeline.py
│   ├── predict.py
│   └── utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Carolljo/Predictive-Maintenance-Intelligence-Platform.git
cd Predictive-Maintenance-Intelligence-Platform
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset

The project uses the AI4I-PMDI dataset. The raw dataset is stored in:

```text
data/raw/AI4I-PMDI.csv
```

The cleaned dataset used by the training pipeline is stored in:

```text
data/processed/cleaned_data.csv
```

## Running the Training Pipeline

The complete model training workflow can be executed from the project root using:

```bash
python -m src.pipeline
```

The training pipeline automatically:

1. Loads the cleaned dataset.
2. Applies feature engineering.
3. Creates a stratified training and test split.
4. Fits the preprocessing pipeline on the training data.
5. Trains the Random Forest classifier.
6. Evaluates the model using accuracy, macro F1, weighted F1, a classification report, and a confusion matrix.
7. Saves the trained model, preprocessing pipeline, and label encoder.

After successful training, the following artifacts are generated:

```text
artifacts/
├── best_model.pkl
├── preprocessor.pkl
└── label_encoder.pkl
```

These artifacts allow the same trained model, preprocessing transformations, and target-label mapping to be reused during inference without retraining the model.

## Running the Prediction Module

After training the model and generating the required artifacts, the prediction module can be tested from the project root using:

```bash
python -m src.predict
```

If the trained artifacts are available, the module loads:

```text
artifacts/
├── best_model.pkl
├── preprocessor.pkl
└── label_encoder.pkl
```

A successful artifact-loading check displays:

```text
Prediction artifacts loaded successfully.
```

The prediction workflow reuses the saved preprocessing pipeline and trained model, ensuring that new machine data undergoes the same transformations used during model training.

## Making Predictions

The trained pipeline can perform predictions on new machine sensor data without retraining the model.

During inference, the system:

1. Applies the same feature engineering used during training.
2. Transforms the input using the saved preprocessing pipeline.
3. Generates a failure prediction using the trained Random Forest model.
4. Converts the encoded prediction back to the original diagnostic label.
5. Returns the predicted failure type along with the model confidence score.

Example prediction output:

```text
Predicted_Failure  Confidence
No failure         1.0
```

This ensures that training and inference use the same feature transformations and target-label mappings.

## Running the Explainability Module

The trained Random Forest model can be inspected using the reusable explainability module.

Run the module from the project root:

```bash
python -m src.explainability
```

Example output:

```text
Top 10 Important Features
-------------------------
                    Feature  Importance
           num__Power_Index    0.217537
           num__Torque (Nm)    0.153011
num__Rotational speed (rpm)    0.147015
       num__Tool wear (min)    0.127829
num__Temperature_Difference    0.085373
                num__System    0.048186
                  num__Year    0.042969
                   num__Day    0.031415
   num__Air temperature (K)    0.025920
                 num__Month    0.021085
```

The module uses the fitted preprocessing pipeline to recover the transformed feature names and matches them with the Random Forest feature-importance scores.

These importance values indicate which features the trained model relied on most strongly when making predictions. They should not be interpreted as evidence that those features directly cause machine failures.

## Limitations and Future Improvements

Although the final Random Forest model achieves high overall accuracy, the project has several important limitations.

### Current Limitations

* **Severe class imbalance:** Most observations belong to the `No failure` class, while some failure categories contain very few samples.
* **Minority-class detection:** The two rarest classes in the test set achieved zero recall, showing that high overall accuracy does not represent equally strong performance across all failure types.
* **Limited failure examples:** Some diagnostic classes contain too few observations for the model to reliably learn their patterns.
* **Dataset-specific patterns:** Date-derived features may capture patterns specific to this dataset rather than meaningful physical relationships with machine failure.
* **Model confidence:** Prediction probabilities represent the model's confidence and should not be interpreted as certainty.
* **Offline prediction:** The current implementation performs predictions from Python code and does not yet expose the model through a production API or real-time monitoring system.

### Future Improvements

Future development of the project can include:

* Collecting or incorporating additional observations for rare failure categories.
* Exploring class-weighted and cost-sensitive learning approaches.
* Comparing additional models designed for imbalanced classification.
* Performing more robust validation using techniques such as stratified cross-validation.
* Investigating probability calibration for more reliable confidence estimates.
* Further validating the usefulness of date-derived features.
* Extending model explainability for individual predictions.
* Building a REST API for real-time prediction requests.
* Containerizing the application using Docker.
* Deploying the prediction service to AWS.
* Adding monitoring for model performance, prediction distributions, and potential data drift.

The current project represents a complete local machine learning workflow, while API development, cloud deployment, and production monitoring remain future stages of the system.


## Technology Stack

The current implementation uses the following technologies and libraries:

* **Python** – core programming language used throughout the project.
* **pandas** – data loading, cleaning, manipulation, and feature engineering.
* **NumPy** – numerical operations.
* **scikit-learn** – preprocessing pipelines, model training, evaluation, and Random Forest classification.
* **imbalanced-learn** – SMOTE experiments for handling class imbalance.
* **Matplotlib** and **Seaborn** – exploratory data analysis and visualization.
* **SHAP** – model explainability experiments.
* **Jupyter Notebook** – exploratory analysis and model experimentation.
* **Git and GitHub** – version control and project repository management.
* **VS Code** – primary development environment.

### Planned Technologies

The next development stages are expected to introduce:

* **FastAPI** – exposing the trained model through a REST API.
* **Docker** – containerizing the prediction service.
* **AWS** – cloud deployment and integration of the predictive maintenance workflow.

## Workflow Summary

The project follows a notebook-to-production-code workflow.

1. **Data understanding and cleaning** are performed in the initial notebooks.
2. **Exploratory data analysis** is used to investigate sensor behaviour, distributions, and failure patterns.
3. **Feature engineering** creates additional machine and date-based variables.
4. **Model experiments** compare baseline models, hyperparameter tuning, and SMOTE.
5. **Model explainability** investigates the features influencing the selected model.
6. The validated workflow is converted into reusable modules inside the `src` package.
7. Running `python -m src.pipeline` executes the complete training workflow and saves the required artifacts.
8. The saved artifacts are reused by `src.predict` for inference without retraining the model.

This structure keeps experimentation separate from reusable application code while maintaining consistency between model training and prediction.
