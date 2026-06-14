# Credit Risk Prediction

This solution transforms raw transaction data into a training-ready dataset and exposes a pre-trained credit risk model through a FastAPI service.

# Setup

Install dependencies:

pip install -r requirements.txt

Generate training dataset:

python data_prep/prepare_data.py

Run API:

uvicorn api.app:app --reload

Run tests:

pytest -v


# Solution Overview

The supplied solution provided a basic implementation for generating a training dataset and exposing a trained model through a FastAPI endpoint. My focus was to improve the solution from a data engineering and production-readiness perspective while maintaining compatibility with the supplied model artifact.

The work completed can be grouped into three areas:

1. Data preparation and feature engineering
2. API enhancements
3. Testing, logging and operational readiness


# Part 1 - Data Engineering

## Data Exploration and Quality Assessment

Before feature engineering, the source datasets were validated for missing values, duplicate transaction IDs, and duplicate customer IDs. These checks help prevent silent data quality issues and ensure the pipeline fails fast when invalid data is encountered

### Findings

No missing values or duplicate records were identified. CUST_0005 was noted as a potential outlier due to a significantly higher credit amount, but was retained given the small sample size.

### Assumptions

- Transaction IDs are unique.
- Each customer has a single label record.
- Positive amounts represent credits and negative amounts represent debits.
- The supplied model is fixed and should remain compatible with the generated features.

## Feature Engineering

Transactions were aggregated to a single customer-level record and the following features were generated:

- txn_count
- total_debit
- total_credit
- avg_amount
- net_cashflow (`total_credit + total_debit`)

`net_cashflow` was added as an additional feature to provide a simple measure of the customer's overall cash position.

## Text Processing

Transaction descriptions were cleaned by converting text to lowercase, removing punctuation and numeric characters, and normalising whitespace. Keyword-based behavioural features were then generated from merchant descriptions such as rent, netflix, tesco, payroll, bonus to capture simple spending and income patterns.

## Training Dataset Creation

Customer-level features were merged with the labels dataset to create a training-ready dataset.

An intermediate text aggregation column (all_desc) was used during feature generation but removed from the final dataset because it is not required by the model and would unnecessarily increase dataset size.

The final dataset is written to: artifacts/training_set.csv


## Exploratory Data Analysis

I generated two exploratory visualisations to validate the engineered features and better understand the sample data.

Visualisations include:

- Transaction count distribution
- Total credit distribution

These charts were used to validate feature generation and highlight potential outliers.

## Logging

Structured logging was added throughout the data preparation process as below

- Start and completion events
- Input record counts
- Validation results
- Number of generated customer records
- Output file location

This improves traceability and operational support.

# Part 2 - API Development

The supplied FastAPI application was enhanced to better align with the requirements and improve operational visibility.

## Improvements Made

### Request and Response Alignment

The API was updated to include customer_id in both the request and response payloads, allowing predictions to be easily traced back to the originating customer.

### Improved Response Handling

Prediction probabilities are explicitly cast to float to avoid JSON serialisation issues.

### Logging

Logging was added for model loading and prediction requests, improving observability and supporting troubleshooting in a production environment.

### Health Endpoint

A /health endpoint was added to support application health checks and readiness monitoring.


# Testing

Basic automated tests were added to validate the health and prediction endpoints, ensuring successful responses and the expected API output structure.

# Part 3: Documentation 

## 1. What part of the exercise did you find most challenging, and why?

The most challenging aspect was balancing completeness with simplicity. I focused on delivering a maintainable and production-ready solution that met the requirements without introducing unnecessary complexity.

## 2. What tradeoffs did you make?

I prioritised simplicity, readability, and maintainability over building a more complex solution.
Key trade-offs included:

- Using keyword-based feature extraction instead of more advanced NLP techniques.
- Implementing lightweight data validation rather than a dedicated data quality framework.
- Preserving compatibility with the supplied model rather than retraining it with additional features.
- Choosing a simple and cost-effective deployment approach aligned with the expected workload.

## 3. Azure Deployment (£500/month, <100ms latency, 1000 predictions/hour)

Given the relatively low traffic volume (~1000 predictions/hour) and lightweight Logistic Regression model, I would deploy the FastAPI service to Azure App Service with Application Insights for monitoring. The model artifact could be packaged with the application or loaded from Azure Blob Storage at startup. This approach would comfortably meet the latency requirement (<100ms) while remaining well within the £500/month budget.

## 4. How would you deploy the FastAPI service and make the model artifact available?

I would containerise the FastAPI application using Docker and deploy it to Azure App Service. For this exercise, I would package the model artifact within the container image to keep deployment simple. If model updates became more frequent, I would store the artifact in Azure Blob Storage and load it during application startup. CI/CD pipelines would automate build, test, and deployment activities.

## 5. If transaction volume increased from thousands to millions per day, how would you rethink Part 1?

The current pandas-based implementation is suitable for prototyping but would not scale efficiently to millions of transactions per day. At larger volumes, I would store data in Azure Data Lake Storage, use Spark for distributed processing, store data in Parquet format, and implement incremental processing to avoid reprocessing historical data. This would improve scalability, performance, and cost efficiency.

## 6. What metrics would you track in production and why?

### API Metrics

- API latency: ensure response times meet SLA requirements.
- Request volume: monitor usage patterns and capacity needs.
- Error rates: identify application or infrastructure issues.

### Data Quality Metrics

- Missing values: detect incomplete data.
- Duplicate records: prevent incorrect predictions.
- Feature distribution changes: identify unusual data patterns.

### Model Metrics

- Prediction distribution: monitor whether predictions remain reasonable.
- Default rates: compare actual outcomes against model predictions.
- Model drift: identify when model performance starts to decline.

Potential risks include model drift, changing customer behaviour, upstream data quality issues and model performance degradation over time.

## 7. AI Tool Usage

ChatGPT was used to review implementation approaches, discuss FastAPI patterns, generate boilerplate examples and assist with documentation. All suggestions were reviewed, tested and adapted before inclusion in the final solution.