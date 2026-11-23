# SCSE3040 MLOps Mini Project --- Group G054 (MP025)

## Problem Overview: Fishing Boat Catch Prediction (Lucknow)
- **Course**: SCSE3040 MLOps
- **Group ID**: G054
- **Problem ID**: MP025
- **Client**: Lucknow Fisherfolk Cooperative
- **Target**: `catch_kg` (Regression, evaluated via Mean Absolute Error)

## Group Members
| Enrolment No. | Name | GitHub Username |
| :--- | :--- | :--- |
| S24CSEU2763 | Jayant Kriplani | jayantkriplani |
| S24CSEU2728 | Aditya Sharma | AdityaSharma |
| S24CSEU2765 | Saranya | saranya. |
| S24CSEU2800 | Nainika Singh | NainikaSingh |
| S24CSEU2780 | Abhibhav Manhas | AbhibhavManhas |

## Project Architecture & Pipeline
- **Data Management**: DVC tracking raw training data (`data/train.csv.dvc`).
- **Data Engineering**: Data cleaning, schema validation, and leak prevention in `src/mp025/data_clean.py`.
- **Feature Engineering**: Modular `ColumnTransformer` pipeline in `src/mp025/features.py`.
- **Tracking & Registry**: MLflow tracking with SQLite backend (`mlflow.db`).
- **Quality Assurance**: Pytest test suite in `tests/`.
