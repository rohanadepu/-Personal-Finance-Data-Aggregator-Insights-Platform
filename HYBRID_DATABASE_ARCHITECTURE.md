# Hybrid Database Architecture - PostgreSQL + DuckDB

## Overview

This Personal Finance Dashboard implements a **hybrid database architecture** that combines the strengths of both PostgreSQL and DuckDB:

- **PostgreSQL**: ACID-compliant transactional database for data integrity and consistency
- **DuckDB**: High-performance analytical database for sub-second query response times

## Architecture Benefits

### PostgreSQL (OLTP)
- **ACID Compliance**: Ensures data consistency and integrity
- **Concurrent Access**: Handles multiple users and applications safely
- **Backup & Recovery**: Enterprise-grade data protection
- **Complex Transactions**: Supports multi-table operations with rollback capability

### DuckDB (OLAP)
- **Sub-second Analytics**: Optimized for analytical queries
- **Columnar Storage**: Efficient for aggregations and reporting
- **In-Process**: No network overhead, embedded in application
- **SQL Compatible**: Standard SQL interface with advanced analytics functions

## Data Flow

```
[Raw Data] → [Airflow DAG] → [PostgreSQL] → [Data Sync] → [DuckDB] → [Streamlit Dashboard]
```

1. **Data Ingestion**: Airflow DAGs process raw transaction data
2. **OLTP Storage**: Clean data stored in PostgreSQL for ACID compliance
3. **Analytics Sync**: Data synchronized to DuckDB for fast analytics
4. **Dashboard**: Streamlit queries DuckDB for real-time insights

## Implementation Details

### Database Schema

#### PostgreSQL Tables
```sql
CREATE TABLE transactions (
    tx_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMP,
    amount DECIMAL(15,2),
    currency VARCHAR(3),
    merchant VARCHAR(255),
    mcc VARCHAR(4),
    account_id VARCHAR(50),
    city VARCHAR(100),
    state VARCHAR(2),
    channel VARCHAR(20),
    category VARCHAR(50),
    subcategory VARCHAR(50),
    is_recurring BOOLEAN,
    normalized_amount DECIMAL(15,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### DuckDB Analytics
- Same schema as PostgreSQL for consistency
- Additional indexes for performance:
  - `idx_timestamp` on timestamp column
  - `idx_merchant` on merchant column  
  - `idx_category` on category column

### Components

#### 1. Database Initialization (`init_db.py`)
- Creates PostgreSQL tables
- Loads sample data for testing
- Sets up indexes for performance

#### 2. Data Sync Utility (`data_sync.py`)
- Syncs data from PostgreSQL to DuckDB
- Validates data consistency
- Provides analytics queries
- Can be run manually or scheduled

#### 3. Streamlit Dashboard (`app.py`)
- Automatically syncs data on load (cached for 5 minutes)
- Falls back to sample data if PostgreSQL unavailable
- Queries DuckDB for fast visualization
- Handles error cases gracefully

## Usage

### Setup

1. **Start Infrastructure**:
   ```bash
   cd infra/compose
   docker compose up -d
   ```

2. **Initialize Database**:
   ```bash
   cd dashboards
   python init_db.py
   ```

3. **Run Dashboard**:
   ```bash
   cd dashboards
   streamlit run app.py
   ```

### Data Sync Management

```bash
# Full sync with validation and stats
python data_sync.py

# Sync only
python data_sync.py --sync

# Validate sync
python data_sync.py --validate

# Show statistics
python data_sync.py --stats

# Custom analytics query
python data_sync.py --query "SELECT category, AVG(amount) FROM transactions GROUP BY category"
```

## Performance Characteristics

### PostgreSQL Queries
- **Write Operations**: ~100-1000 TPS
- **Complex Joins**: ~50-500ms
- **Concurrent Users**: 10-100s

### DuckDB Queries  
- **Aggregations**: ~1-10ms
- **Analytical Queries**: ~10-100ms
- **Data Scanning**: ~100MB/s per core

## Monitoring & Maintenance

### Health Checks
1. **PostgreSQL Connection**: `python init_db.py`
2. **Data Sync Status**: `python data_sync.py --validate`
3. **Dashboard Access**: Visit `http://localhost:8501`

### Scheduled Sync
Add to crontab for regular sync:
```bash
# Sync every 5 minutes
*/5 * * * * cd /path/to/dashboards && python data_sync.py --sync
```

### Backup Strategy
- **PostgreSQL**: Use `pg_dump` for full backups
- **DuckDB**: Rebuild from PostgreSQL using sync utility

## Error Handling

The system handles various failure scenarios:

1. **PostgreSQL Unavailable**: Falls back to sample data
2. **DuckDB Corruption**: Rebuilds from PostgreSQL
3. **Network Issues**: Cached data serves dashboard
4. **Schema Changes**: Sync utility recreates DuckDB tables

## Resume Benefits

This implementation demonstrates:

- **Database Architecture Design**: Multi-database system design
- **Performance Optimization**: OLTP/OLAP separation
- **Data Engineering**: ETL pipeline with Airflow
- **Error Handling**: Graceful fallback mechanisms
- **Monitoring**: Health checks and validation
- **Documentation**: Comprehensive system documentation

## Future Enhancements

1. **Real-time Sync**: Change data capture (CDC) from PostgreSQL
2. **Partitioning**: Time-based partitioning for large datasets
3. **Caching Layer**: Redis for session data
4. **API Layer**: REST API for external access
5. **Machine Learning**: Integration with ML models for predictions
