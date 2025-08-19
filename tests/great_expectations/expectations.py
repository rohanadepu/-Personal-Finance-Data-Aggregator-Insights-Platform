from great_expectations.core import ExpectationConfiguration, ExpectationSuite
from great_expectations.dataset import PandasDataset
import pandas as pd

def create_transaction_expectations():
    suite = ExpectationSuite(expectation_suite_name="transaction_suite")
    
    expectations = [
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "tx_id"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "timestamp"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={"column": "amount"}
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "amount",
                "min_value": -1000000,  # Adjust based on your data
                "max_value": 1000000
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "currency",
                "value_set": ["USD", "EUR", "GBP"]
            }
        ),
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column": "mcc",
                "regex": "^[0-9]{4}$"
            }
        )
    ]
    
    for expectation in expectations:
        suite.add_expectation(expectation)
    
    return suite

def validate_transactions(df):
    dataset = PandasDataset(df)
    suite = create_transaction_expectations()
    
    results = dataset.validate(expectation_suite=suite)
    return results

if __name__ == "__main__":
    # Example usage
    import duckdb
    
    # Load sample data
    conn = duckdb.connect(database=':memory:', read_only=False)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    
    # Validate data
    validation_results = validate_transactions(df)
    print(f"Validation success: {validation_results.success}")
    
    # Print failed expectations
    for result in validation_results.results:
        if not result.success:
            print(f"Failed: {result.expectation_config.kwargs['column']} - {result.expectation_config.expectation_type}")
