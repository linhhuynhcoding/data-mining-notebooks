# clean/video_cleaner.py
from __future__ import annotations
import io, os, tempfile, uuid
from datetime import datetime, timezone
from typing import List, Dict

import cv2
import imagehash
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class VideoCleaner:
    """
    - Lọc video quá ngắn/dài, hỏng
    - Chống trùng bằng perceptual hash (pHash) khung đầu tiên, với ngưỡng Hamming (near-duplicate)
    - Lưu:
        cleaned mp4 -> cleaned/videos/{domain}/{keyword}/...
        cleaned metadata parquet -> metadata/cleaned/videos/{domain}/{batch_id}.parquet
        duplicates parquet       -> metadata/cleaned/videos/{domain}/{batch_id}_duplicates.parquet
        pHash index per-domain   -> metadata/cleaned/videos/index/{domain}/phash_index.parquet
    - Trả: DataFrame các bản ghi KHÔNG TRÙNG để pipeline chạy tiếp
    """
    def __init__(
        self,
        minio_client,
        bucket: str,
        min_duration: float = 1.0,
        max_duration: float = 60.0,
        phash_threshold: int = 4,  # 0..64, càng nhỏ càng chặt
    ):
        self.client = minio_client
        self.bucket = bucket
        self.min_d = float(min_duration)
        self.max_d = float(max_duration)
        self.phash_threshold = int(phash_threshold)

    # ---------- MinIO helpers ----------
    def _get_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, key)
        try:
            return resp.read()
        finally:
            resp.close(); resp.release_conn()

    def _put_bytes(self, key: str, b: bytes, content_type: str = "video/mp4"):
        self.client.put_object(self.bucket, key, io.BytesIO(b), length=len(b), content_type=content_type)

    def _try_get(self, key: str) -> bytes | None:
        try:
            resp = self.client.get_object(self.bucket, key)
            try:
                return resp.read()
            finally:
                resp.close(); resp.release_conn()
        except Exception:
            return None

    # ---------- pHash index (per-domain) ----------
    def _index_key(self, domain: str) -> str:
        return f"metadata/cleaned/videos/index/{domain}/phash_index.parquet"

    def _load_index(self, domain: str) -> pd.DataFrame:
        b = self._try_get(self._index_key(domain))
        if not b:
            return pd.DataFrame(columns=["phash64", "object_key_cleaned"])
        return pd.read_parquet(io.BytesIO(b))

    def _save_index(self, domain: str, df: pd.DataFrame):
        key = self._index_key(domain)
        bio = io.BytesIO()
        df.to_parquet(bio, index=False)
        self._put_bytes(key, bio.getvalue(), content_type="application/octet-stream")

    # ---------- video utils ----------
    @staticmethod
    def _duration_sec(path: str) -> float:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        return (frames / fps) if fps > 0 else 0.0

    @staticmethod
    def _first_frame(path: str) -> Image.Image | None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _phash64(img: Image.Image) -> int:
        ph = imagehash.phash(img)  # 64-bit perceptual hash
        return int(str(ph), 16)

    @staticmethod
    def _ham(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    # ---------- main ----------
    def clean_batch(self, meta_records: List[Dict]) -> pd.DataFrame:
        if not meta_records:
            return pd.DataFrame([])

        # gom theo domain để dùng index riêng
        by_domain: dict[str, list[dict]] = {}
        for m in meta_records:
            by_domain.setdefault(m["domain"], []).append(m)

        kept_rows: list[dict] = []

        for domain, items in by_domain.items():
            index_df = self._load_index(domain)
            known = index_df["phash64"].astype("Int64").dropna().astype(int).tolist() if len(index_df) else []
            new_index_rows = []
            dup_rows = []

            batch_id = str(uuid.uuid4())
            when = datetime.now(timezone.utc).isoformat()

            for m in items:
                raw_key = m["raw_path"]

                # tải file tạm
                b = self._get_bytes(raw_key)
                fd, tmp = tempfile.mkstemp(suffix=".mp4")
                try:
                    os.write(fd, b)
                finally:
                    os.close(fd)

                try:
                    # validate duration
                    dur = self._duration_sec(tmp)
                    if dur < self.min_d or dur > self.max_d:
                        dup_rows.append({**m, "drop_reason": "duration", "duration": float(dur), "when": when})
                        continue

                    # pHash khung đầu
                    pil = self._first_frame(tmp)
                    if pil is None:
                        dup_rows.append({**m, "drop_reason": "decode_fail", "when": when})
                        continue

                    ph64 = self._phash64(pil)

                    # near-duplicate check
                    is_dup = any(self._ham(ph64, old) <= self.phash_threshold for old in known)
                    if is_dup:
                        dup_rows.append({**m, "drop_reason": f"phash<= {self.phash_threshold}", "when": when})
                        continue

                    # accept -> upload cleaned (rehost)
                    cleaned_key = raw_key.replace("raw/videos/", "cleaned/videos/")
                    self._put_bytes(cleaned_key, b, content_type="video/mp4")

                    out = dict(m)
                    out.update({
                        "cleaned_path": cleaned_key,
                        "duration": float(dur),
                        "phash64": int(ph64),
                    })
                    kept_rows.append(out)

                    known.append(ph64)
                    new_index_rows.append({"phash64": int(ph64), "object_key_cleaned": cleaned_key})

                finally:
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

            # ghi manifest cleaned & duplicates cho batch (nếu có)
            df_keep = pd.DataFrame([r for r in kept_rows if r["domain"] == domain])
            if not df_keep.empty:
                key = f"metadata/cleaned/videos/{domain}/{batch_id}.parquet"
                bio = io.BytesIO()
                df_keep.to_parquet(bio, index=False)
                self._put_bytes(key, bio.getvalue(), content_type="application/octet-stream")

            if dup_rows:
                df_dup = pd.DataFrame(dup_rows)
                key = f"metadata/cleaned/videos/{domain}/{batch_id}_duplicates.parquet"
                bio = io.BytesIO()
                df_dup.to_parquet(bio, index=False)
                self._put_bytes(key, bio.getvalue(), content_type="application/octet-stream")

            # cập nhật index
            if new_index_rows:
                new_df = pd.DataFrame(new_index_rows)
                index_df = pd.concat([index_df, new_df], ignore_index=True)
                index_df = index_df.drop_duplicates(subset=["phash64"], keep="first")
                self._save_index(domain, index_df)

        return pd.DataFrame(kept_rows)
