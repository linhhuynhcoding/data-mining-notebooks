import io
from PIL import Image
import pandas as pd

class ImageCleaner:
    def __init__(self, minio_client, bucket: str, min_size=64):
        self.client = minio_client
        self.bucket = bucket
        self.min_size = min_size

    def _get_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        return data

    def _put_bytes(self, key: str, b: bytes, content_type: str="image/jpeg"):
        self.client.put_object(self.bucket, key, io.BytesIO(b), length=len(b), content_type=content_type)

    def _valid_and_normalize(self, b: bytes) -> bytes | None:
        try:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            w, h = img.size
            if w < self.min_size or h < self.min_size:
                return None
            bio = io.BytesIO()
            img.save(bio, format="JPEG", quality=92)
            return bio.getvalue()
        except Exception:
            return None

    def clean_batch(self, meta_records: list[dict]) -> pd.DataFrame:
        rows = []
        for m in meta_records:
            raw_key = m["raw_path"]
            b = self._get_bytes(raw_key)
            nb = self._valid_and_normalize(b)
            if nb is None:
                continue
            cleaned_key = raw_key.replace("raw/images/", "cleaned/images/").rsplit(".",1)[0] + ".jpg"
            self._put_bytes(cleaned_key, nb, content_type="image/jpeg")
            m2 = dict(m)
            m2["cleaned_path"] = cleaned_key
            rows.append(m2)
        return pd.DataFrame(rows)
