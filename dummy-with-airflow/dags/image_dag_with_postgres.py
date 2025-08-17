from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
import uuid
import os

DATA_DIR = "/root/coding/data-mining-notebooks/data"

def fetch_keywords_from_db():
    hook = PostgresHook(postgres_conn_id="postgres_image")
    records = hook.get_records("SELECT id, keyword, target_count FROM keyword_config")
    return [{"keyword_id": r[0], "keyword": r[1], "target": r[2]} for r in records]

def crawl_images(**context):
    keywords = fetch_keywords_from_db()
    hook = PostgresHook(postgres_conn_id="postgres_image")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for kw in keywords:
        batch_id = str(uuid.uuid4())
        images = [{"url": f"http://example.com/{i}.jpg"} for i in range(kw["target"])]
        file_path = f"{DATA_DIR}/{batch_id}_raw.parquet"
        pd.DataFrame(images).to_parquet(file_path)

        hook.run(
            "INSERT INTO batch_metadata (keyword_id, batch_id, status, file_path, total_images) VALUES (%s, %s, %s, %s, %s)",
            parameters=(kw["keyword_id"], batch_id, "new", file_path, len(images))
        )
    

def validate_images(**context):
    hook = PostgresHook(postgres_conn_id="postgres_image")
    batches = hook.get_records("SELECT batch_id, file_path FROM batch_metadata WHERE status='new'")
    for batch_id, file_path in batches:
        df = pd.read_parquet(file_path)
        df["is_valid"] = True  # giả lập validate
        new_path = file_path.replace("_raw", "_cleaned")
        df.to_parquet(new_path)
        hook.run("UPDATE batch_metadata SET status='cleaned', file_path=%s WHERE batch_id=%s", parameters=(new_path, batch_id))

def preprocess_images(**context):
    hook = PostgresHook(postgres_conn_id="postgres_image")
    batches = hook.get_records("SELECT batch_id, file_path FROM batch_metadata WHERE status='cleaned'")
    for batch_id, file_path in batches:
        df = pd.read_parquet(file_path)
        df["processed"] = True  # giả lập preprocess
        new_path = file_path.replace("_cleaned", "_processed")
        df.to_parquet(new_path)
        hook.run("UPDATE batch_metadata SET status='processed', file_path=%s WHERE batch_id=%s", parameters=(new_path, batch_id))

with DAG(
    dag_id="image_pipeline_prod",
    start_date=datetime(2025, 8, 12),
    schedule_interval="@daily",
    catchup=False
) as dag:

    task_extract = PythonOperator(
        task_id="crawl_images",
        python_callable=crawl_images
    )

    task_validate = PythonOperator(
        task_id="validate_images",
        python_callable=validate_images
    )

    task_preprocess = PythonOperator(
        task_id="preprocess_images",
        python_callable=preprocess_images
    )

    task_extract >> task_validate >> task_preprocess
