#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-time loader to push cleaned Steam review data into Postgres.

This script:
  * Reads the three Excel files defined in train_steam_model.DATA_FILES
  * Runs train_steam_model.prepare_dataset to create the binary target, etc.
  * Writes the resulting DataFrame into a Postgres table steam_reviews_prepared
    in your DigitalOcean Postgres database.

Requirements:
  * APP_DB_URL must be set in your environment or .env file.
    Example (DigitalOcean):
      APP_DB_URL=postgresql://doadmin:<password>@<host>:25060/defaultdb?sslmode=require
"""

import os

import pandas as pd
from sqlalchemy import create_engine

from .train_steam_model import DATA_FILES, load_all_files, prepare_dataset, load_dotenv

# Make sure .env is loaded (in case this script is run directly)
load_dotenv()

DB_URL = os.environ.get("APP_DB_URL")


def main() -> None:
    if not DB_URL:
        raise RuntimeError(
            "APP_DB_URL is not set. Please add it to your .env with your "
            "DigitalOcean Postgres connection string."
        )

    print("Loading raw Excel files...")
    df_raw = load_all_files(DATA_FILES)

    print("Preparing dataset...")
    df_prepared = prepare_dataset(df_raw)

    print("Connecting to Postgres...")
    engine = create_engine(DB_URL)

    table_name = "steam_reviews_prepared"
    print(f"Writing {len(df_prepared)} rows to table {table_name!r}...")
    df_prepared.to_sql(table_name, engine, if_exists="replace", index=False)

    print("Done writing to Postgres!")


if __name__ == "__main__":
    main()