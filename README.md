# Risk Modeling & Loan Default Prediction Platform

An end-to-end credit risk analytics platform built on **2M+ Lending Club loans**, predicting probability of default, risk tier classification, and expected financial loss. Designed to demonstrate ML Engineering, Data Engineering, and MLOps skills at a fintech production level.

---

## Business Problem

Financial institutions lose billions annually to loan defaults. This platform provides:

1. **Probability of Default (PD)** — Binary classification: what is the likelihood this borrower will default?
2. **Risk Tier Classification** — Multi-class: Low / Medium / High / Very High Risk
3. **Expected Loss Estimation** — Regression: `Expected Loss = PD × Exposure × LGD`

These three models support lending decisions, portfolio risk management, and regulatory capital estimation.

---

## Architecture

```
Raw CSV (1.6 GB)
     │
     ▼
Data Validation ──► Schema checks, null thresholds, type validation
     │
     ▼
Parquet Storage ──► Columnar format, Snappy compression
     │
     ▼
PostgreSQL ──────► Analytical queries, portfolio aggregations
     │
     ▼
Feature Store ───► 25+ engineered financial features (Parquet)
     │
     ▼
Training Pipeline ► 9 models × 3 objectives, SMOTE resampling
     │
     ▼
Model Registry ──► MLflow tracking, versioned artifacts
     │
     ▼
FastAPI Service ─► /predict, /health, /model-info, /metrics
     │
     ▼
Streamlit Dashboard ► Risk assessment + Portfolio analytics
     │
     ▼
Evidently Monitoring ► Data drift, feature drift detection
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Lending Club (Kaggle)](https://www.kaggle.com/datasets/wordsforthewise/lending-club) |
| Records | 2,260,668 loans |
| Features | 151 original columns |
| Time Range | 2007–2018 |
| Target | `loan_status`: Charged Off (default) vs Fully Paid |
| Default Rate | ~20.5% |

### Why Lending Club?
- **Scale**: 2M+ real financial records — not a toy dataset
- **Complexity**: 150+ features including credit history, employment, geography
- **Real-world**: Actual loan outcomes, not synthetic data
- **Industry-relevant**: Consumer lending is the core use case for credit risk models

---

## Model Results

### Baseline Model Comparison (220K training rows, 37 application-time features)

| Model | ROC-AUC | F1 Score | Train Time |
|-------|---------|----------|------------|
| **CatBoost** | **0.7386** | **0.4552** | 2.1s |
| GradientBoosting | 0.7378 | 0.2321 | 226s |
| LightGBM | 0.7360 | 0.4531 | 2.3s |
| XGBoost | 0.7336 | 0.4529 | 1.1s |
| RandomForest | 0.7317 | 0.4491 | 5.3s |
| ExtraTrees | 0.7301 | 0.4466 | 1.6s |
| LogisticRegression | 0.7189 | 0.4463 | 3.0s |
| DecisionTree | 0.7071 | 0.4341 | 2.1s |
| NaiveBayes | 0.6734 | 0.3439 | 0.0s |

### Optuna Hyperparameter Tuning (40 trials per model)

| Model | Baseline AUC | Tuned AUC | Improvement |
|-------|-------------|-----------|-------------|
| **CatBoost** | 0.7386 | **0.7401** | +0.20% |
| LightGBM | 0.7360 | 0.7396 | +0.49% |
| XGBoost | 0.7336 | 0.7395 | +0.80% |

**Deployed production model**: **CatBoost (Optuna-tuned, AUC=0.7401)** — highest AUC,
best F1 (0.455), and **100× faster** than GradientBoosting. Models use native class
weighting (`scale_pos_weight` / `auto_class_weights="Balanced"`) rather than SMOTE,
yielding better-calibrated probabilities and intuitive SHAP explanations.

### Sanity check — model discriminates as expected

| Applicant | Grade | DTI | Int Rate | **Predicted PD** | Risk Tier |
|-----------|-------|-----|----------|------------------|-----------|
| Subprime | G | 32% | 26% | **95.3%** | Very High Risk |
| Near-prime | D | 18% | 14% | **59.9%** | Very High Risk |
| Prime | A | 6% | 6% | **3.1%** | Low Risk |


### Risk Scoring Output

| Risk Tier | Count | Avg Risk Score | Avg PD | Total Expected Loss |
|-----------|-------|---------------|--------|-------------------|
| Low Risk | 2,096 | 817 | 6% | $989K |
| Medium Risk | 3,466 | 755 | 17% | $4.59M |
| High Risk | 2,531 | 656 | 35% | $8.53M |
| Very High Risk | 708 | 523 | 59% | $4.75M |

---

## Feature Engineering

25+ engineered features across 5 categories:

### Financial Ratios
- `loan_income_ratio` — Loan amount relative to annual income
- `debt_income_ratio` — DTI normalized to 0-1 scale
- `installment_income_ratio` — Monthly payment burden
- `revolving_utilization_ratio` — Credit limit usage
- `interest_payment_ratio` — Interest cost relative to income

### Risk Scores
- `financial_health_score` — Composite score (DTI, utilization, delinquency, LTI)
- `credit_strength_score` — Credit profile strength indicator
- `repayment_capacity_score` — Ability to service debt

### Credit Features
- `total_credit_lines_ratio` — Open vs total accounts
- `delinquency_rate` — Historical payment failures
- `inquiry_frequency` — Recent credit-seeking behavior

### Interaction Features
- `income × credit_strength` — Income weighted by credit quality
- `dti × loan_amount` — Debt burden amplified by loan size
- `int_rate × loan_amount` — Total interest exposure

---

## Explainability

### SHAP Analysis
- **Top risk factors**: Interest rate, sub-grade, revolving utilization, DTI
- Global feature importance via SHAP summary and bar plots
- Local explanations for individual loan decisions

### Business Explanation Generator
Human-readable output for each prediction:
```
Decision: ELEVATED RISK — Recommend Review
Probability of Default: 27.3%
Risk Score: 621
Expected Loss: $4,914

Top Risk Factors:
  1. High Interest Rate (18.5%)
  2. Elevated Debt-to-Income Ratio (28.3)
  3. High Revolving Utilization (82%)

Top Positive Factors:
  1. Strong Employment Stability (8 years)
  2. Good Credit Account History (25 accounts)
```

---

## Project Structure

```
loan-risk-prediction/
├── data/
│   ├── raw/                     # Original Lending Club CSVs
│   ├── processed/               # Cleaned Parquet files
│   └── feature_store/           # Engineered features
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory Data Analysis
├── src/
│   ├── config.py                # Project configuration
│   ├── data_ingestion.py        # CSV → Parquet pipeline
│   ├── data_validation.py       # Schema & quality checks
│   ├── database.py              # PostgreSQL operations
│   ├── feature_engineering.py   # 25+ financial features
│   ├── feature_store.py         # Feature persistence layer
│   ├── feature_selection.py     # Mutual info, permutation importance
│   ├── preprocessing.py         # Imputation, encoding, scaling
│   ├── training.py              # 9 models + MLflow tracking
│   ├── evaluation.py            # Metrics, ROC, PR, Lift curves
│   ├── explainability.py        # SHAP, LIME, business explanations
│   ├── risk_scoring.py          # Credit risk score engine (300-850)
│   ├── drift_detection.py       # Evidently monitoring
│   └── utils.py                 # Logging, plotting helpers
├── app/
│   ├── api.py                   # FastAPI prediction service
│   └── dashboard.py             # Streamlit analytics dashboard
├── models/                      # Serialized model artifacts
├── reports/figures/             # Generated plots & charts
├── sql/
│   ├── create_tables.sql        # PostgreSQL schema
│   └── queries.sql              # Analytical SQL queries
├── tests/
│   ├── test_features.py         # Feature engineering tests
│   ├── test_preprocessing.py    # Preprocessing tests
│   └── test_api.py              # API endpoint tests
├── deploy/
│   ├── terraform/               # AWS infrastructure as code
│   │   ├── main.tf              # VPC, ECS, RDS, ALB, S3, ECR
│   │   ├── variables.tf         # Configurable parameters
│   │   └── outputs.tf           # Deployment URLs & endpoints
│   └── scripts/
│       ├── build_and_push.sh    # Build & push Docker → ECR
│       ├── upload_model.sh      # Upload model → S3
│       └── deploy.sh            # Full deployment orchestrator
├── .github/workflows/ci.yml    # GitHub Actions CI/CD + AWS deploy
├── Dockerfile                   # Multi-stage (api + dashboard targets)
├── docker-compose.yml           # Local multi-service deployment
├── dvc.yaml                     # Data versioning pipeline
├── run_pipeline.py              # One-command full pipeline
├── requirements.txt             # Python dependencies
└── setup.py                     # Package configuration
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- Kaggle CLI configured

### Setup
```bash
# Clone and setup
git clone https://github.com/yourusername/loan-risk-prediction.git
cd loan-risk-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download dataset
kaggle datasets download -d wordsforthewise/lending-club -p data/raw/ --unzip

# Run full pipeline
python run_pipeline.py
```

### Run Services
```bash
# FastAPI
uvicorn app.api:app --reload --port 8000

# Streamlit Dashboard
streamlit run app/dashboard.py

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

### Docker
```bash
docker-compose up --build
```

### API Usage
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "annual_income": 65000,
    "loan_amount": 15000,
    "dti": 18.0,
    "int_rate": 12.0,
    "installment": 450.0,
    "emp_length_years": 5,
    "revol_util": 45.0,
    "revol_bal": 12000,
    "open_acc": 8,
    "total_acc": 20,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "pub_rec": 0,
    "grade_numeric": 2,
    "sub_grade_numeric": 10
  }'
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python, SQL |
| **ML Models** | CatBoost, XGBoost, LightGBM, scikit-learn |
| **Feature Engineering** | Custom Feature Store, Target Encoding |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **Database** | PostgreSQL |
| **Backend API** | FastAPI |
| **Frontend** | Streamlit |
| **Monitoring** | Evidently AI |
| **Deployment** | Docker, AWS ECS Fargate, Terraform |
| **Storage** | S3 (model artifacts) |
| **CI/CD** | GitHub Actions |
| **Explainability** | SHAP, LIME |

---

## AWS Deployment

### Architecture

```
Internet → ALB (port 80)
               │
               ├── /predict, /health, /docs  →  ECS Fargate (API container)
               │                                      │
               └── /* (default)              →  ECS Fargate (Dashboard container)
                                                      │
                                                 RDS PostgreSQL (private subnet)
                                                      │
                                                 S3 (model artifacts, versioned)
```

### Infrastructure (Terraform)

All AWS resources are defined in `deploy/terraform/`:
- **VPC** with public/private subnets across 2 AZs
- **ECS Fargate** cluster with API and Dashboard services
- **RDS PostgreSQL 15** (db.t3.micro) in private subnet
- **ALB** with path-based routing (`/predict` → API, `/*` → Dashboard)
- **ECR** repositories for Docker images
- **S3** bucket with versioning for model artifacts
- **CloudWatch** log groups for container monitoring
- **IAM** roles with least-privilege (ECS task gets S3 read/write only)

### Deploy from Scratch

```bash
# 1. Provision infrastructure
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your db_password
terraform init
terraform plan
terraform apply

# 2. Build and push images to ECR
cd ../..
bash deploy/scripts/build_and_push.sh <aws-account-id> us-east-1

# 3. Upload trained model to S3
bash deploy/scripts/upload_model.sh credit-risk-model-artifacts

# 4. Or do it all at once
bash deploy/scripts/deploy.sh <aws-account-id> us-east-1
```

### CI/CD Auto-Deploy

Pushes to `main` automatically:
1. Run tests (pytest + ruff)
2. Build Docker images (multi-stage: api + dashboard)
3. Push to ECR (tagged with commit SHA + latest)
4. Force new ECS deployment
5. Wait for service stability

Requires GitHub secrets: `AWS_ROLE_ARN` (OIDC role for GitHub Actions).

### Cost Estimate (us-east-1)

| Resource | Spec | Monthly Cost |
|----------|------|-------------|
| ECS Fargate (API) | 0.5 vCPU, 1 GB | ~$15 |
| ECS Fargate (Dashboard) | 0.25 vCPU, 0.5 GB | ~$8 |
| RDS PostgreSQL | db.t3.micro, 20 GB | ~$15 |
| ALB | Standard | ~$16 |
| S3 | < 1 GB | ~$0.03 |
| ECR | < 2 GB | ~$0.20 |
| **Total** | | **~$54/month** |

---

## Future Improvements

- [x] ~~Deploy to AWS (ECS + RDS) with Terraform~~ ✅
- [ ] Add real-time feature computation with Apache Kafka
- [ ] Implement A/B testing framework for model versions
- [ ] Add customer segmentation with K-Means clustering
- [ ] Build default forecasting time series model
- [ ] Add Grafana monitoring dashboard
- [ ] Implement model retraining triggers on drift detection

---

## License

MIT License
