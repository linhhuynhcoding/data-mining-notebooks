import os, sys, time, uuid
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

# thêm sys.path để import module bên ngoài /dags
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from minio import Minio

from ingest.image_fetcher import ImageFetcher
from ingest.video_fetcher import VideoFetcher
from clean.image_cleaner import ImageCleaner
from clean.video_cleaner import VideoCleaner
from preprocess.image_preprocessor import ImagePreprocessor
from preprocess.video_preprocessor import VideoPreprocessor
from feature.image_feature_extractor import ImageFeatureExtractor
from feature.video_feature_extractor import VideoFeatureExtractor
from utils import fetch_keywords_from_db, update_crawl_task
# ---------- ENV ----------
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")
MINIO_HOST     = os.getenv("MINIO_HOST", "https://minio.nhatlinh.ddns.net")
MINIO_ACCESS   = os.getenv("MINIO_ACCESS_KEY", "nhatlinh")
MINIO_SECRET   = os.getenv("MINIO_SECRET_KEY", "nhatlinh")
MINIO_SECURE   = os.getenv("MINIO_SECURE", "true").lower() == "true"
BUCKET         = os.getenv("MINIO_BUCKET", "datalake")
DATASOURCE = os.getenv("DATASOURCE", "")

# 10 keyword cho domain 'technology'
# KEYWORDS = [
#     "artificial intelligence",
#     "machine learning",
#     "quantum computing",
#     "cybersecurity",
#     "robotics technology",
#     "virtual reality",
#     "augmented reality",
#     "blockchain technology",
#     "internet of things",
#     "data center servers",
# ]

MAX_IMAGES = 100
TOTAL_IMAGES = 100
TOTAL_VIDEOS = 50

def create_minio_client() -> Minio:
    return Minio(MINIO_HOST, access_key=MINIO_ACCESS, secret_key=MINIO_SECRET, secure=MINIO_SECURE)

# ---------- EXTRACT ----------
def extract_images(**context):
    client = create_minio_client()
    fetcher = ImageFetcher(PEXELS_API_KEY, client, BUCKET)
    total_fetched = 0
    keywords = fetch_keywords_from_db()

    # per_kw = max(1, (TOTAL_IMAGES + len(KEYWORDS) - 1) // len(KEYWORDS))

    records = []

    for kw in keywords:
        remaining_global = MAX_IMAGES - total_fetched
        if remaining_global <= 0:
            break

        remaining_kw = kw["target_count"] - kw["current_count"]
        if remaining_kw <= 0:
            continue

        allowed = min(remaining_kw, remaining_global)

        # ✅ pass current_page (last_page_fetched + 1) to fetch
        current_page = (kw["last_page_fetched"] or 0) + 1
        df, page = fetcher.fetch(
            domain=kw["domain"],
            keyword=kw["keyword"],
            target=allowed,
            current_page=current_page
        )

        if df is not None and len(df):
            rows = df.to_dict(orient="records")
            records.extend(rows)
            total_fetched += len(rows)

            # update DB with new count + page
            update_crawl_task(
                kw,
                new_count=kw["current_count"] + len(rows),
                new_page=page
            )

        if total_fetched >= MAX_IMAGES:
            break
    rows = rows[:TOTAL_IMAGES]
    context["ti"].xcom_push(key="raw_image_meta", value=rows)

# def extract_videos(**context):
#     """
#     Crawl ~50 video technology (chia đều 10 keyword), xử lý rate limit,
#     upload RAW vào MinIO và trả metadata qua XCom.
#     """
#     client = create_minio_client()
#     fetcher = VideoFetcher(PEXELS_API_KEY, client, BUCKET)
#     per_kw = max(1, (TOTAL_VIDEOS + len(KEYWORDS) - 1) // len(KEYWORDS))

#     records = []
#     for kw in KEYWORDS:
#         df = fetcher.fetch(domain="technology", keyword=kw, target=per_kw)
#         if df is not None and len(df):
#             records.extend(df.to_dict(orient="records"))
#         if len(records) >= TOTAL_VIDEOS:
#             break

#     records = records[:TOTAL_VIDEOS]
#     context["ti"].xcom_push(key="raw_video_meta", value=records)

# ---------- CLEAN ----------
def clean_images(**context):
    client = create_minio_client()
    cleaner = ImageCleaner(client, BUCKET)
    meta = context["ti"].xcom_pull(key="raw_image_meta", task_ids="extract.extract_images")
    if not meta: return
    df = cleaner.clean_batch(meta)
    context["ti"].xcom_push(key="cleaned_image_meta", value=df.to_dict(orient="records"))

# def clean_videos(**context):
#     client = create_minio_client()
#     cleaner = VideoCleaner(client, BUCKET, min_duration=1.0, max_duration=60.0)
#     meta = context["ti"].xcom_pull(key="raw_video_meta", task_ids="extract.extract_videos")
#     if not meta:
#         return
#     df = cleaner.clean_batch(meta)
#     context["ti"].xcom_push(key="cleaned_video_meta", value=df.to_dict(orient="records"))

# ---------- PREPROCESS ----------
def preprocess_images(**context):
    client = create_minio_client()
    pre = ImagePreprocessor(client, BUCKET, out_size=(224,224), save_npy=True)
    meta = context["ti"].xcom_pull(key="cleaned_image_meta", task_ids="clean.clean_images")
    if not meta: return
    df = pre.preprocess_batch(meta)
    context["ti"].xcom_push(key="preprocessed_image_meta", value=df.to_dict(orient="records"))

# def preprocess_videos(**context):
#     client = create_minio_client()
#     pre = VideoPreprocessor(client, BUCKET, frame_rate=1.0, out_size=(224,224), max_frames=64)
#     meta = context["ti"].xcom_pull(key="cleaned_video_meta", task_ids="clean.clean_videos")
#     if not meta:
#         return
#     df = pre.preprocess_batch(meta)
#     context["ti"].xcom_push(key="preprocessed_video_meta", value=df.to_dict(orient="records"))

# ---------- FEATURES ----------
def extract_image_features(**context):
    client = create_minio_client()
    extractor = ImageFeatureExtractor(device="cpu")
    meta = context["ti"].xcom_pull(key="preprocessed_image_meta", task_ids="preprocess.preprocess_images")
    if not meta: return
    # Ghi Parquet shards vào features/image/resnet18/... (xem class cho prefix)
    extractor.extract_batch(client, BUCKET, meta)

# def extract_video_features(**context):
#     client = create_minio_client()
#     extractor = VideoFeatureExtractor(device="cpu")
#     meta = context["ti"].xcom_pull(key="preprocessed_video_meta", task_ids="preprocess.preprocess_videos")
#     if not meta:
#         return
#     _ = extractor.extract_batch(client, BUCKET, meta)

# ---------- DAG ----------
default_args = {
    "owner": "team2",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
}

with DAG(
    dag_id="data_ingestion_pipeline",
    start_date=datetime(2025, 8, 1),
    schedule_interval="*/10 * * * *",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    description="Crawl (Pexels) -> Clean -> Preprocess -> Feature (images & videos) -> MinIO",
    tags=["data-mining", "pexels", "minio"],
) as dag:
    with TaskGroup("extract") as extract_group:
        t1 = PythonOperator(task_id="extract_images", python_callable=extract_images)
        # t2 = PythonOperator(task_id="extract_videos", python_callable=extract_videos)

    with TaskGroup("clean") as clean_group:
        c1 = PythonOperator(task_id="clean_images", python_callable=clean_images)
        # c2 = PythonOperator(task_id="clean_videos", python_callable=clean_videos)

    with TaskGroup("preprocess") as preprocess_group:
        p1 = PythonOperator(task_id="preprocess_images", python_callable=preprocess_images)
        # p2 = PythonOperator(task_id="preprocess_videos", python_callable=preprocess_videos)

    with TaskGroup("features") as feature_group:
        f1 = PythonOperator(task_id="extract_image_features", python_callable=extract_image_features)
        # f2 = PythonOperator(task_id="extract_video_features", python_callable=extract_video_features)

    # extract_group >> clean_group >> preprocess_group >> feature_group
    extract_group >> clean_group >> preprocess_group 
