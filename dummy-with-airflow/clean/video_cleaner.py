import io, tempfile, os, cv2, pandas as pd

class VideoCleaner:
    def __init__(self, minio_client, bucket: str, min_duration=1.0, max_duration=60.0):
        self.client = minio_client
        self.bucket = bucket
        self.min_d = min_duration
        self.max_d = max_duration

    def _get_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        return data

    def _put_bytes(self, key: str, b: bytes, content_type="video/mp4"):
        self.client.put_object(self.bucket, key, io.BytesIO(b), length=len(b), content_type=content_type)

    def _duration_sec(self, path: str) -> float:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        return (n / fps) if fps > 0 else 0.0

    def clean_batch(self, meta_records: list[dict]) -> pd.DataFrame:
        rows = []
        for m in meta_records:
            raw_key = m["raw_path"]
            data = self._get_bytes(raw_key)
            # tạm lưu để OpenCV đọc duration
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                f.write(data)
                temp = f.name
            dur = self._duration_sec(temp)
            os.unlink(temp)
            if dur < self.min_d or dur > self.max_d:
                continue

            cleaned_key = raw_key.replace("raw/videos/", "cleaned/videos/")
            self._put_bytes(cleaned_key, data, content_type="video/mp4")
            m2 = dict(m)
            m2["cleaned_path"] = cleaned_key
            m2["duration"] = dur
            rows.append(m2)
        return pd.DataFrame(rows)
