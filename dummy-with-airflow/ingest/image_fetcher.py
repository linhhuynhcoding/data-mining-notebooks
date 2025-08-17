import pandas as pd
import uuid
import requests
import os
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
from airflow.providers.postgres.hooks.postgres import PostgresHook
from urllib.parse import quote, urlencode
from datetime import datetime


API_URL = 'https://api.pexels.com/v1/search'
DOMAINS = ["River in Mountain Forest", "Green Tree Forest", "sea water"]
SAVE_DIR = '../coding/data-mining-notebooks/data/raw/image'
DATA_DIR = "../coding/data-mining-notebooks/data/metadata"
load_dotenv()
API_KEY = os.getenv("API_KEY")

def fetch_keywords_from_db():
    hook = PostgresHook(postgres_conn_id="postgres_image")
    records = hook.get_records("""
        SELECT id, keyword, target_count, current_count, last_page_fetched, domain
        FROM crawl_tasks
        WHERE status IS NULL OR current_count < target_count
        FOR UPDATE SKIP LOCKED
    """)
    return [{"keyword_id": r[0], "keyword": r[1], "target_count": r[2], "current_count": r[3], "last_page_fetched": r[4], "domain": r[5]} for r in records]
        
def download_image(url: str, save_dir: str, prefix: str):
    try:
        os.makedirs(save_dir, exist_ok=True)

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        filename = os.path.basename(prefix + "_" + urlparse(url).path)
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"✅ Đã lưu: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Lỗi với {url}: {e}")
        return ""

def fetch(domain, keyword, page, batch_id):
    try:
        response = requests.get(API_URL + "?query=" + quote(keyword) +"&per_page=" + str(30) + "&page=" + str(page), timeout=10, headers={"Authorization": API_KEY})
        print("?query=" + quote(keyword) +"&per_page=" + str(30) + "&page=" + str(page))
        response.raise_for_status()
        # response.headers.get("X-Ratelimit-Limit")
        data = response.json()  # Giả sử API trả về danh sách URL ảnh
    except Exception as e:
        print(f"Lỗi khi gọi API: {e}")
        exit()
        
    photos = data.get("photos", [])
    print(f"🔍 Tìm thấy {len(photos)} ảnh với query")
    
    images = []
    
    for photo in photos:
        img_url = photo.get("src", {})
        if img_url:
            image_dir = SAVE_DIR + "/" + domain.replace(" ", "_") + "/" + keyword.replace(" ", "_") 
            filepath = download_image(img_url.get("original", {}), image_dir, batch_id)
            images.append({
                "image_id": photo.get("id", int),
                "original_url": img_url,
                "original_width": photo.get("width", int),
                "original_height": photo.get("height", int),
                "local_path": filepath,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "domain": domain,
                "keyword": keyword
            })
        else:
            print(f"⚠️ Bỏ qua ảnh thiếu URL: {img_url}")
            
    return {
        "total": len(images),
        "page": page,
        "images": images
    }
                
def api_fetch_images(**ctx):
    keywords = fetch_keywords_from_db()
    hook = PostgresHook(postgres_conn_id="postgres_image")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    results = []

    for kw in keywords:
        batch_id = str(uuid.uuid4())
        
        data = fetch(kw["domain"], kw["keyword"], kw["last_page_fetched"] + 1, batch_id)
        
        file_path = f"{DATA_DIR}/{kw["domain"]}/{kw["keyword"]}"
        os.makedirs(file_path, exist_ok=True)
        filepath = os.path.join(file_path, f"{batch_id}_raw.parquet")
        pd.DataFrame(data["images"]).to_parquet(filepath)

        hook.run(
            """
            INSERT INTO crawl_tasks (
                id, domain, keyword, target_count, current_count, last_page_fetched, status, batch_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, 'new', %s
            )
            ON CONFLICT (id) DO UPDATE
            SET current_count = EXCLUDED.current_count,
                last_page_fetched = EXCLUDED.last_page_fetched,
                batch_id = EXCLUDED.batch_id,
                status = 'new',
                updated_at = now();
            """,
            parameters=(kw["keyword_id"], kw["domain"], kw["keyword"], kw["target_count"], kw["current_count"] + data["total"], kw["last_page_fetched"] + 1, batch_id)
        )
        
        results.append({
            "batch_id": batch_id,
            "images": data["images"],
            "domain": kw["domain"],
            "keyword": kw["keyword"]
        })
    
    ctx['ti'].xcom_push(key="results", value=results)
    
    # return results
        

