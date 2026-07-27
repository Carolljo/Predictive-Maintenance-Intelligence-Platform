# Predictive Maintenance Intelligence Platform

A machine learning project for predicting machine failure types from operational and sensor data.

I started this project as a notebook-based ML workflow and gradually converted it into a reusable prediction system. The final version includes a Random Forest model, reusable preprocessing and inference code, a local FastAPI interface, and an AWS deployment using SageMaker, Lambda, API Gateway, SNS, S3, CloudWatch, and IAM.

The main goal was not just to train a model with high accuracy, but to understand the full workflow from messy data to a working cloud prediction API.

---

## What the Project Does

The system takes machine information such as temperature, rotational speed, torque, tool wear, machine type, and timestamp data and predicts one of six diagnostic conditions:

- No failure
- Heat Dissipation Failure
- Overstrain Failure
- Power Failure
- Tool Wear Failure
- Random Failures

For the cloud deployment, a client sends machine data to an API Gateway endpoint. Lambda validates the request and sends it to a SageMaker real-time endpoint for prediction.

If the model predicts a failure, Lambda publishes an SNS notification. If the prediction is `No failure`, no alert is sent.

---

## System Architecture

![Predictive Maintenance AWS Architecture](docs/architecture.png)

The deployed request flow is:

```text
Client
  ↓
API Gateway
  ↓
Lambda
  ↓
SageMaker Endpoint
  ↓
Random Forest Prediction
  ↓
Lambda
  ├── Failure → SNS → Email Alert
  └── No failure → No Alert
  ↓
JSON Response
```

S3 stores the packaged model artifacts used when the SageMaker model is deployed. CloudWatch records Lambda executions, while IAM controls communication between the AWS services.

The model itself was trained locally. SageMaker is used here for managed real-time inference rather than model training.

---

## Dataset

The project uses the AI4I-PMDI predictive-maintenance dataset with **10,000 machine observations**.

The main inputs are:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type
- System
- Control
- Date

The target is `Diagnostic`.

### Class Distribution

| Diagnostic Class | Samples |
|---|---:|
| No failure | 9,652 |
| Heat Dissipation Failure | 106 |
| Overstrain Failure | 98 |
| Power Failure | 83 |
| Tool Wear Failure | 42 |
| Random Failures | 19 |

One of the biggest challenges in this project is the class imbalance. More than 96% of the observations belong to `No failure`, while some failure classes contain fewer than 50 examples.

Because of that, I did not rely only on accuracy when evaluating the model.

---

## Data Preparation

The original data contained a large amount of missing sensor information, so I kept the raw and processed datasets separate.

```text
data/raw/AI4I-PMDI.csv
data/processed/cleaned_data.csv
```

During data understanding and cleaning I checked:

- missing values
- data types
- duplicates
- unusual sensor values
- feature distributions
- outliers
- target imbalance

The raw dataset is kept unchanged so the cleaning process remains reproducible.

---

## Feature Engineering

I added two machine-related features:

### Temperature Difference

```text
Process temperature - Air temperature
```

This represents the temperature difference between the machine process and the surrounding environment.

### Power Index

```text
Rotational speed × Torque
```

This is a simple engineered measure of machine operating load.

`Power_Index` later became the most important feature in the Random Forest feature-importance analysis.

The `Date` column is also converted into:

```text
Year
Month
Day
Day_of_Week
Quarter
```

The original timestamp is removed after these values are generated.

This is also why production prediction requests still require `Date`: the inference code needs it to recreate the same features that were used during training.

---

## Preprocessing

I built the preprocessing stage using a scikit-learn `ColumnTransformer`.

Numerical features use:

```text
Median Imputation
        ↓
StandardScaler
```

Categorical features use:

```text
Most-Frequent Imputation
        ↓
OneHotEncoder
```

The preprocessor is fitted only on the training data and then saved.

The same fitted preprocessor is reused during inference instead of rebuilding preprocessing separately.

---

## Model Development

I compared several classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Gradient Boosting
- XGBoost

Random Forest gave the best balance for the final workflow and was selected as the production model.

I also experimented with Random Forest hyperparameter tuning and SMOTE.

Neither experiment gave enough improvement to justify replacing the baseline Random Forest, so I kept them as experiments instead of automatically using the more complicated approach.

---

## Model Performance

The final model was evaluated on a stratified 20% test split.

| Metric | Score |
|---|---:|
| Accuracy | **0.9925** |
| Macro F1 | **0.6572** |
| Weighted F1 | **0.9897** |

The accuracy looks excellent at first, but it needs context.

The model performs extremely well on the dominant `No failure` class and several failure classes, but the rarest classes remain difficult to detect. Two minority classes had zero recall on the test split.

So I would **not** describe this model as "99.25% reliable at detecting machine failures."

The more important weakness is visible in the Macro F1 score of **0.6572**.

That class-imbalance problem is one of the main limitations of the current model.

---

## Explainability

I used Random Forest feature importance and experimented with SHAP to better understand what the model was learning.

The most important production-model features included:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | Power Index | 0.2175 |
| 2 | Torque | 0.1530 |
| 3 | Rotational speed | 0.1470 |
| 4 | Tool wear | 0.1278 |
| 5 | Temperature Difference | 0.0854 |
| 6 | System | 0.0482 |

These values show which features the Random Forest used most heavily. They should not be interpreted as proof that those features cause machine failures.

The SHAP experiments are available in:

```text
notebooks/09_Model_Explainability.ipynb
```

---

## From Notebooks to Reusable Code

The first part of the project was developed through notebooks for data exploration and experimentation.

After the modelling stage, I moved the main workflow into reusable modules under `src/`.

```text
src/
├── config.py
├── data_loader.py
├── explainability.py
├── feature_engineering.py
├── model_evaluator.py
├── model_trainer.py
├── pipeline.py
├── predict.py
├── preprocessing.py
└── utils.py
```

The training pipeline can be run with:

```bash
python -m src.pipeline
```

It handles:

```text
Data Loading
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
Preprocessing
    ↓
Random Forest Training
    ↓
Evaluation
    ↓
Artifact Saving
```

The saved production artifacts are:

```text
artifacts/
├── best_model.pkl
├── preprocessor.pkl
└── label_encoder.pkl
```

---

## Local FastAPI Interface

Before connecting the project to AWS, I created a small FastAPI interface for local prediction testing.

Run it from the project root with:

```bash
uvicorn app.main:app --reload
```

The prediction route is:

```text
POST /predict
```

FastAPI/Pydantic validates the request and converts it into the structure expected by the prediction pipeline.

This local API is mainly useful for development and testing. The deployed cloud API uses API Gateway and Lambda instead.

---

## AWS Deployment

The cloud version uses:

- **Amazon S3** for packaged model storage
- **Amazon SageMaker** for real-time model inference
- **AWS Lambda** for validation and orchestration
- **Amazon API Gateway** for the HTTP prediction endpoint
- **Amazon SNS** for failure email alerts
- **Amazon CloudWatch** for Lambda logs
- **AWS IAM** for service permissions

### SageMaker

The model artifacts are packaged into `model.tar.gz` and uploaded to S3.

The deployment code is in:

```text
deployment/sagemaker/
├── deploy.py
└── inference.py
```

`inference.py` recreates the same feature-engineering and preprocessing workflow used during training before making a prediction.

The endpoint can be deployed with:

```bash
python deployment/sagemaker/deploy.py
```

I used a separate SageMaker environment because the deployed model artifacts were created with scikit-learn 1.4.2 and needed a compatible deployment environment.

The tested deployment environment is documented in:

```text
requirements-sagemaker.txt
```

### Lambda and API Gateway

Lambda receives the API Gateway request, validates the required fields and invokes the SageMaker endpoint.

The production route is:

```text
POST /predict
```

A successful response looks like:

```json
{
  "predicted_failure": "Power Failure",
  "confidence": 1.0,
  "alert_sent": true
}
```

If SageMaker returns a failure class, Lambda sends the machine information to SNS.

For a healthy prediction:

```json
{
  "predicted_failure": "No failure",
  "confidence": 1.0,
  "alert_sent": false
}
```

No SNS alert is sent.

---

## End-to-End Test

I tested both branches of the deployed system using actual records from the processed dataset.

### Failure Test

A known `Power Failure` record produced:

```text
Prediction: Power Failure
Confidence: 1.0
SNS alert: Sent
Email: Received
```

### No-Failure Test

A known healthy record produced:

```text
Prediction: No failure
Confidence: 1.0
SNS alert: Not sent
```

This verified the complete path:

```text
API Gateway
→ Lambda
→ SageMaker
→ Prediction
→ Conditional SNS Alert
→ API Response
```

CloudWatch logs were also checked to confirm successful Lambda executions.

---

## Project Structure

```text
Predictive-Maintenance-Intelligence-Platform/
│
├── app/
│   └── main.py
│
├── artifacts/
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   └── preprocessor.pkl
│
├── data/
│   ├── raw/
│   │   └── AI4I-PMDI.csv
│   └── processed/
│       ├── cleaned_data.csv
│       └── engineered_data.csv
│
├── deployment/
│   └── sagemaker/
│       ├── deploy.py
│       └── inference.py
│
├── docs/
│   └── architecture.png
│
├── lambda_app/
│   └── lambda_function.py
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
│   ├── config.py
│   ├── data_loader.py
│   ├── explainability.py
│   ├── feature_engineering.py
│   ├── model_evaluator.py
│   ├── model_trainer.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── utils.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-sagemaker.txt
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/Carolljo/Predictive-Maintenance-Intelligence-Platform.git
cd Predictive-Maintenance-Intelligence-Platform
```

Create the normal development environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\Activate.ps1
```

Install the local project dependencies:

```bash
pip install -r requirements.txt
```

Run the training pipeline:

```bash
python -m src.pipeline
```

Run model explainability:

```bash
python -m src.explainability
```

For the tested SageMaker deployment environment, the project uses **Python 3.11.4** with the versions recorded in:

```text
requirements-sagemaker.txt
```

---

## Limitations

The biggest limitation is class imbalance.

The model is very strong at recognising normal operation but is not equally reliable across all failure categories. More minority-class data would be needed before treating this as a serious industrial failure-detection system.

There are a few other limitations I would investigate before production use:

- The date-derived features may contain dataset-specific patterns rather than meaningful physical relationships.
- Random Forest probabilities are not automatically calibrated, so a confidence score of `1.0` should not be treated as absolute certainty.
- The API currently demonstrates the prediction architecture but does not include production authentication or rate limiting.
- Model and prediction drift are not monitored.
- AWS infrastructure is created manually rather than through infrastructure-as-code.

The SageMaker endpoint is also intentionally deleted when I am not testing the project because a real-time endpoint continues to generate charges while provisioned.

---

## What I Would Improve Next

The next modelling priority would be improving minority-class detection rather than chasing a slightly higher overall accuracy.

I would explore:

- class-weighted and cost-sensitive models
- stratified cross-validation
- additional minority-class data
- probability calibration
- stronger validation of the date features
- per-prediction SHAP explanations

For the deployment side, the next improvements would be API authentication, model versioning, drift monitoring, CI/CD, infrastructure-as-code, and externalising AWS configuration such as endpoint names and SNS topic information.

---

## Tech Stack

**Machine Learning:** Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, SHAP

**API:** FastAPI, Pydantic, Uvicorn

**AWS:** S3, SageMaker, Lambda, API Gateway, SNS, CloudWatch, IAM, Boto3

**Development:** Jupyter Notebook, VS Code, Git, GitHub, AWS CLI

---

## Project Status

**v1.0 is complete.**

The current version covers the full path from raw machine data and model experimentation to reusable ML code and a tested AWS inference workflow.

The cloud endpoint is not kept permanently online because of SageMaker endpoint costs, but it can be recreated from the deployment code and stored model artifacts.

---

## Author

**Carol**

GitHub: Carolljo

This project was developed as part of my Data Science internship and focuses on building an end-to-end predictive maintenance workflow, from data preparation and machine learning to cloud-based inference on AWS.