import io, os, time, uuid, requests
from datetime import datetime, timezone
import pandas as pd

class VideoFetcher:
    API_URL = "https://api.pexels.com/videos/search"

    def __init__(self, api_key: str, minio_client, bucket: str):
        self.api_key = api_key
        self.client = minio_client
        self.bucket = bucket

    def _put_bytes(self, key: str, b: bytes, content_type: str = "video/mp4"):
        self.client.put_object(
            self.bucket, key, io.BytesIO(b), length=len(b),
            content_type=content_type
        )

    def fetch(self, domain: str, keyword: str, target: int, per_page: int = 15) -> pd.DataFrame:
        headers = {"Authorization": self.api_key}
        rows = []
        page = 1
        count = 0
        batch_id = str(uuid.uuid4())

        while count < target:
            params = {"query": keyword, "per_page": per_page, "page": page}
            r = requests.get(self.API_URL, headers=headers, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(60)
                continue
            r.raise_for_status()
            data = r.json()
            vids = data.get("videos", [])
            if not vids:
                break

            for v in vids:
                if count >= target:
                    break
                files = v.get("video_files", [])
                mp4 = next((f for f in files if f.get("file_type","").startswith("video/mp4")), None)
                if not mp4:
                    continue
                url = mp4.get("link")
                if not url:
                    continue
                got = requests.get(url, timeout=120)
                if got.status_code == 429:
                    time.sleep(60)
                    continue
                got.raise_for_status()
                content = got.content

                fname = f"{batch_id}_{v['id']}.mp4"
                key = f"raw/videos/{domain}/{keyword}/{fname}"
                self._put_bytes(key, content, content_type="video/mp4")

                rows.append({
                    "id": str(v["id"]),
                    "domain": domain,
                    "keyword": keyword,
                    "raw_path": key,
                    "width": v.get("width"),
                    "height": v.get("height"),
                    "duration": v.get("duration"),
                    "url": url,
                    "batch_id": batch_id,
                    "crawl_time": datetime.now(timezone.utc).isoformat()
                })
                count += 1

            page += 1

        df = pd.DataFrame(rows)
        if len(df):
            out_key = f"metadata/raw/videos/{domain}/{batch_id}.parquet"
            bio = io.BytesIO()
            df.to_parquet(bio, index=False)
            self._put_bytes(out_key, bio.getvalue(), content_type="application/octet-stream")
        return df
