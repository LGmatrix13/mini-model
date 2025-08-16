from abc import ABC, abstractmethod
import polars as pl
from sqlalchemy import Engine, create_engine
import logging


class MiniModelBase(ABC):
    def __init__(self, verbose: bool):
        super().__init__()
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
    def _log_info(self, message: str):
        if self.verbose: self.logger.info(message)
    @abstractmethod
    def ingest(self) -> pl.DataFrame:
        """Provide the dataframe you wish to serve for training"""
        self._log_info("Ingesting data")
        raise NotImplementedError("ingest not yet implemented")
    @abstractmethod
    def embedder(self, item: str) -> list[float]:
        """Handle embedding string columns in your dataset"""
        raise NotImplementedError("ingest not yet implemented")
    @abstractmethod
    def postgres(self) -> Engine:
        """Return SQLAlchemy-compatible Postgres enginge."""
        self._log_info("Connecting to postgres")
        raise NotImplementedError("postgres not setup")
    def _process(self, batch_size: int = 500):
        """
        Process and stream-write a Polars DataFrame to Postgres in batches.
        """
        self._log_info("Processing data")
        df = self.ingest()
        str_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
        engine = self.postgres()
        total_rows = len(df)
        self._log_info(f"Proceeding with {total_rows // batch_size} batches")
        batch = 1
        for start in range(0, total_rows, batch_size):
            batch_df = df.slice(start, batch_size)
            batch_df = batch_df.with_columns([
                pl.col(col).apply(
                    lambda item: self.embedder(item) if item is not None else None
                ).alias(f"{col}_embedding")
                for col in str_cols
            ])
            batch_pd = batch_df.to_pandas()
            batch_pd.to_sql(
                "minimodel_processed",
                engine,
                if_exists="append",
                index=False
            )
            self._log_info(f"Processed and peristed batch {batch}")
            batch += 1
        self._log_info("Processed data successfully")

            
    
    