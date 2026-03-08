# California Housing — End-to-End MLOps Pipeline

## Project Objective

This project demonstrates the design, deployment, and monitoring of an end-to-end Machine Learning solution following MLOps best practices.

The goal is to build a fully automated pipeline capable of:

• Training a machine learning model
• Tracking experiments and model versions
• Deploying the model as a production-ready API
• Automating integration and deployment workflows
• Monitoring model performance in production
• Detecting data drift
• Automatically retraining the model when performance degrades

This project simulates real-world industrial AI deployment conditions.

## Business Use Case

We aim to predict California housing prices using socio-economic and geographical data.

Such predictive systems can assist:

• Real estate professionals
• Urban planning institutions
• Investment analysts

by providing fast and automated price estimations.

## Project Architecture Overview

The pipeline follows a complete MLOps lifecycle:

- Data collection and preprocessing
- Model training and evaluation
- Experiment tracking with MLflow
- Model packaging and containerization
- API deployment with FastAPI
- Continuous Integration & Continuous Deployment
- Production monitoring and drift detection
- Automated model retraining

## Technology Stack

Layer	Tools Used
Machine Learning	Scikit-learn, XGBoost
Experiment Tracking	MLflow
API Framework	FastAPI
Containerization	Docker
CI/CD	GitHub Actions
Monitoring	Evidently AI
Retraining	Python automation scripts
Model Versioning	MLflow Model Registry

## API Deployment

The trained model is deployed as a REST API using FastAPI.

### Main Endpoints:

GET /health
Checks if the API is running properly.

POST /predict
Returns a predicted housing price based on input features.

Interactive API documentation is available via Swagger UI:

http://localhost:8000/docs

## Monitoring & Drift Detection

To ensure reliability in production, the system includes:

• Data drift detection
• Performance monitoring
• Alert mechanisms

Drift is detected by comparing:

Reference training dataset

Incoming production data

When significant drift is detected, the retraining pipeline is triggered automatically.

## Automated Retraining Pipeline

If monitoring metrics indicate degraded performance:

✔ New data is collected
✔ Model retraining is triggered
✔ New model version is logged in MLflow
✔ Updated model becomes available for deployment

This ensures continuous model improvement.

## CI/CD Automation

Each code update triggers an automated workflow:

• Docker image build
• API validation tests
• Deployment verification

This guarantees stability and reproducibility of the system.

## Repository Structure

.github/workflows → CI/CD pipelines  
data              → Monitoring datasets  
monitoring        → Drift detection scripts  
retraining        → Automated retraining pipeline  
notebook          → Model training & experiments  
src               → API and ML logic  
Dockerfile        → Container definition

## How to Run the Project Locally

### Clone the repository
git clone https://github.com/your-username/california-housing-mlops-pipeline.git

### Install dependencies
pip install -r requirements.txt

⚠️ Due to dependency conflicts between MLflow (Pydantic v2) and Evidently (Pydantic v1), the project must be run using Docker to ensure environment consistency and reproducibility.

### Run the API
uvicorn src.api:app --reload

### Access API documentation
http://localhost:8000/docs

