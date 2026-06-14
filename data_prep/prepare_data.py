# src/part1_prepare_data.py
from pathlib import Path
import pandas as pd
import re
import logging
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Functions to Clean and Validate Data
# ---------------------------------------------------------------------------

def clean_text(s: str) -> str:
    """Clean and normalize text data by converting to lowercase, removing non-alphabetic characters, and stripping extra whitespace."""
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def validate_dataframes(tx: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Validate the transactions and labels dataframes for missing values and duplicates."""
    # Check for missing values
    if tx.isnull().any().any():
        raise ValueError("Transactions dataframe contains missing values.") 
    if labels.isnull().any().any():
        raise ValueError("Labels dataframe contains missing values.")

    # Check for duplicate transaction IDs
    if tx["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found in transactions dataframe.")

    # Check for duplicate customer IDs in labels
    if labels["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs found in labels dataframe.")

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logger.info("Starting data preparation pipeline")
    
    # Load source datasets
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["txn_timestamp"])
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    logger.info("Loaded %s transactions", len(transactions))
    logger.info("Loaded %s customer labels", len(labels))

    # Validate source data quality
    validate_dataframes(transactions, labels)
    logger.info("Data quality validation completed successfully")

    # Clean transaction descriptions
    transactions["clean_desc"] = transactions["description"].fillna("").apply(clean_text)

    # Generate features by aggregating transaction data at the customer level
    features = (
        transactions.groupby("customer_id")
        .agg(
            txn_count=("transaction_id", "count"),
            total_debit=("amount", lambda x: x[x < 0].sum()),
            total_credit=("amount", lambda x: x[x > 0].sum()),
            avg_amount=("amount", "mean"),
            all_desc=("clean_desc", lambda x: " ".join(x)),
        )
        .reset_index()
    )

    # Additional engineered feature representing customer's overall cash position
    features["net_cashflow"] = (
        features["total_credit"] +
        features["total_debit"]
        )

    # Generate keyword features based on transaction descriptions
    keywords = ["rent", "netflix", "tesco", "payroll", "bonus"]
    for keyword in keywords:
        features[f"kw_{keyword}"] = features["all_desc"].str.contains(rf"\b{keyword}\b").astype(int)

    training_set = features.merge(labels, on="customer_id", how="left")
    
    # Removing all_desc as it is only used internally to create keyword features and is not needed in the final training dataset.
    training_set = training_set.drop(columns=["all_desc"])
    logger.info("Generated features for %s customers", len(training_set))
    

    # ---------------------------------------------------------------------------
    # Exploratory Data Analysis
    # ---------------------------------------------------------------------------

    # Transaction count distribution
    training_set["txn_count"].hist()

    plt.title("Transaction Count Distribution")
    plt.xlabel("Number of Transactions")
    plt.ylabel("Frequency")

    plt.savefig(
        ARTIFACTS_DIR / "txn_count_distribution.png"
        )

    plt.close()

    # Total credit distribution
    training_set["total_credit"].hist()

    plt.title("Total Credit Distribution")
    plt.xlabel("Total Credit")
    plt.ylabel("Frequency")

    plt.savefig(
        ARTIFACTS_DIR / "total_credit_distribution.png"
        )

    plt.close()

    logger.info("EDA visualisations generated successfully")

    # Save the training dataset to artifacts directory
    output_path = ARTIFACTS_DIR / "training_set.csv"
    training_set.to_csv(output_path, index=False)
    logger.info("Training dataset written to %s", output_path)
    logger.info("Data preparation pipeline completed successfully")