import pandas as pd
from sqlalchemy import create_engine, text

from src.config import DB_URL, DB_CONFIG
from src.utils import get_logger

logger = get_logger(__name__)


def get_engine():
    return create_engine(DB_URL)


def create_database():
    admin_url = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/postgres"
    )
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db"),
            {"db": DB_CONFIG["database"]},
        )
        if not result.fetchone():
            conn.execute(text(f"CREATE DATABASE {DB_CONFIG['database']}"))
            logger.info(f"Created database '{DB_CONFIG['database']}'")
        else:
            logger.info(f"Database '{DB_CONFIG['database']}' already exists")
    engine.dispose()


def init_schema():
    from src.config import ROOT_DIR
    sql_path = ROOT_DIR / "sql" / "create_tables.sql"
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(sql_path.read_text()))
        conn.commit()
    logger.info("Schema initialized")
    engine.dispose()


def load_to_postgres(df: pd.DataFrame, table_name: str = "raw_loans",
                     if_exists: str = "replace", chunksize: int = 10000):
    engine = get_engine()
    if if_exists == "replace":
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS predictions CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS feature_store CASCADE"))
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            conn.commit()
    cols_to_load = [c for c in df.columns if c in [
        "loan_amnt", "funded_amnt", "term", "int_rate", "installment",
        "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
        "verification_status", "loan_status", "purpose", "addr_state",
        "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
        "revol_bal", "revol_util", "total_acc", "application_type",
        "tot_cur_bal", "total_rev_hi_lim", "is_default",
    ]]
    df_subset = df[cols_to_load]
    df_subset.to_sql(table_name, engine, if_exists=if_exists,
                     index=False, chunksize=chunksize, method="multi")
    logger.info(f"Loaded {len(df_subset):,} rows into '{table_name}'")
    engine.dispose()


def query_db(sql: str) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(sql, engine)
    engine.dispose()
    return df


def create_supporting_tables():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                loan_id INTEGER,
                model_version VARCHAR(50),
                probability_of_default NUMERIC,
                risk_tier VARCHAR(20),
                expected_loss NUMERIC,
                risk_score INTEGER,
                predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_predictions_tier ON predictions(risk_tier)"
        ))
        conn.commit()
    logger.info("Supporting tables created")
    engine.dispose()


def setup_database(df: pd.DataFrame):
    create_database()
    load_to_postgres(df)
    create_supporting_tables()
    logger.info("Database setup complete")
