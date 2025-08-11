from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from ingest.image_crawler import crawl_images
from ingest.image_fetcher import api_fetch_images
from preprocess.image_preprocess import preprocess_images
from load.loader import upload_to_minio
import requests
import os
from minio import Minio
# from PIL import Image
import io

with DAG(
    dag_id="image_ingestion_pipeline",
    start_date=datetime.now() - timedelta(minutes=1),
    schedule_interval="*/5 * * * *",  # chạy mỗi 5 phút
    catchup=False,
    tags=["image", "minio", "etl"]
) as dag:

    with TaskGroup("extract") as extract_group:
        crawl_web = PythonOperator(
            task_id="crawl_web",
            python_callable=crawl_images
        )

        fetch_api = PythonOperator(
            task_id="fetch_api",
            python_callable=api_fetch_images
        )
        
    with TaskGroup("validate_and_clean") as clean_group:
        validate_image = PythonOperator(
            task_id="validate_image",
            python_callable=""
        )
        
        clean_image = PythonOperator(
            task_id="clean_image",
            python_callable=""
        )

    with TaskGroup("transform") as transform_group:
        preprocess = PythonOperator(
            task_id="preprocess",
            python_callable=preprocess_images
        )

    with TaskGroup("load") as load_group:
        upload_minio = PythonOperator(
            task_id="upload_minio",
            python_callable=upload_to_minio
        )

    # Pipeline
    [crawl_web, fetch_api] >> preprocess >> upload_minio