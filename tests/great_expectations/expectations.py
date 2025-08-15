from great_expectations.core import ExpectationConfiguration, ExpectationSuite
from great_expectations.dataset import SparkDFDataset
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import DataContextConfig

def create_transaction_expectation_suite():
    """Create expectation suite for transaction data"""
    suite = ExpectationSuite(
        expectation_suite_name="transactions_raw_suite"
    )
    
    # Add expectations
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={
                "column_list": [
                    "tx_id",
                    "timestamp",
                    "amount",
                    "currency",
                    "merchant",
                    "mcc",
                    "account_id",
                    "city",
                    "state",
                    "channel"
                ]
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={
                "column": "tx_id"
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={
                "column": "amount"
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_between",
            kwargs={
                "column": "amount",
                "min_value": 0,
                "max_value": 1000000
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "currency",
                "value_set": ["USD", "EUR", "GBP"]
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs={
                "column": "mcc",
                "regex": "^[0-9]{4}$"
            }
        )
    )
    
    return suite

def create_transformed_transaction_suite():
    """Create expectation suite for transformed transaction data"""
    suite = ExpectationSuite(
        expectation_suite_name="transactions_transformed_suite"
    )
    
    # Add expectations for transformed data
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_table_columns_to_match_ordered_list",
            kwargs={
                "column_list": [
                    "tx_id",
                    "timestamp",
                    "amount",
                    "currency",
                    "merchant",
                    "mcc",
                    "account_id",
                    "city",
                    "state",
                    "channel",
                    "category",
                    "subcategory",
                    "is_recurring",
                    "normalized_amount"
                ]
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={
                "column": "category"
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_in_set",
            kwargs={
                "column": "category",
                "value_set": [
                    "Groceries",
                    "Dining",
                    "Transportation",
                    "Shopping",
                    "Entertainment",
                    "Healthcare",
                    "Utilities",
                    "Other"
                ]
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_be_of_type",
            kwargs={
                "column": "is_recurring",
                "type_": "boolean"
            }
        )
    )
    
    suite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs={
                "column": "normalized_amount"
            }
        )
    )
    
    return suite

# Initialize context and save suites
context = BaseDataContext(DataContextConfig())
context.save_expectation_suite(create_transaction_expectation_suite())
context.save_expectation_suite(create_transformed_transaction_suite())
