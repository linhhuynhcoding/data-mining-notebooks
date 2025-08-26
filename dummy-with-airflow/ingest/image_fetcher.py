import io, os, time, uuid, math, requests
from datetime import datetime, timezone
import pandas as pd

class ImageFetcher:
    API_URL = "https://api.pexels.com/v1/search"

    def __init__(self, api_key: str, minio_client, bucket: str):
        self.api_key = api_key
        self.client = minio_client
        self.bucket = bucket

    def _put_bytes(self, key: str, b: bytes, content_type: str = "image/jpeg"):
        self.client.put_object(
            self.bucket, key, io.BytesIO(b), length=len(b),
            content_type=content_type
        )

    def fetch(self, domain: str, keyword: str, target: int, per_page: int = 30) -> pd.DataFrame:
        headers = {"Authorization": self.api_key}
        rows = []
        page = 1
        count = 0
        batch_id = str(uuid.uuid4())

        while count < target:
            params = {"query": keyword, "per_page": per_page, "page": page}
            r = requests.get(self.API_URL, headers=headers, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(60)  # rate limit
                continue
            r.raise_for_status()
            data = r.json()
            photos = data.get("photos", [])
            if not photos:
                break

            for p in photos:
                if count >= target:
                    break
                src = p.get("src", {})
                url = src.get("large") or src.get("original")
                if not url:
                    continue
                # tải bytes
                img = requests.get(url, timeout=60)
                if img.status_code == 429:
                    time.sleep(60)
                    continue
                img.raise_for_status()
                content = img.content

                fname = f"{batch_id}_{p['id']}.jpg"
                key = f"raw/images/{domain}/{keyword}/{fname}"
                self._put_bytes(key, content, content_type="image/jpeg")

                rows.append({
                    "id": str(p["id"]),
                    "domain": domain,
                    "keyword": keyword,
                    "raw_path": key,
                    "width": p.get("width"),
                    "height": p.get("height"),
                    "url": url,
                    "batch_id": batch_id,
                    "crawl_time": datetime.now(timezone.utc).isoformat()
                })
                count += 1

            page += 1

        df = pd.DataFrame(rows)
        if len(df):
            # Lưu parquet metadata theo batch
            out_key = f"metadata/raw/images/{domain}/{batch_id}.parquet"
            bio = io.BytesIO()
            df.to_parquet(bio, index=False)
            self._put_bytes(out_key, bio.getvalue(), content_type="application/octet-stream")

        return df
