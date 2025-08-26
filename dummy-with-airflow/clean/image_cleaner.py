# clean/image_cleaner.py
import io, hashlib, uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import imagehash

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageCleaner:
    """
    - Chuẩn hoá ảnh (RGB, min_size, JPEG)
    - Lọc trùng: sha256 tuyệt đối + pHash (Hamming <= threshold)
    - Ghi:
        cleaned jpg -> cleaned/images/{domain}/{keyword}/xxx.jpg
        cleaned metadata parquet -> metadata/cleaned/images/{domain}/{batch_id}.parquet
        duplicates parquet       -> metadata/cleaned/images/{domain}/{batch_id}_duplicates.parquet
        hash index per-domain    -> metadata/cleaned/images/index/{domain}/hash_index.parquet
    - Trả: DataFrame các bản ghi KHÔNG TRÙNG để pipeline dùng tiếp
    """
    def __init__(self, minio_client, bucket: str, min_size=64, phash_threshold: int = 4):
        self.client = minio_client
        self.bucket = bucket
        self.min_size = int(min_size)
        self.phash_threshold = int(phash_threshold)

    # ---------- MinIO helpers ----------
    def _get_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, key)
        try:
            return resp.read()
        finally:
            resp.close(); resp.release_conn()

    def _put_bytes(self, key: str, b: bytes, content_type: str):
        self.client.put_object(self.bucket, key, io.BytesIO(b), length=len(b), content_type=content_type)

    def _try_get_object(self, key: str) -> bytes | None:
        try:
            resp = self.client.get_object(self.bucket, key)
            try:
                return resp.read()
            finally:
                resp.close(); resp.release_conn()
        except Exception:
            return None

    def _read_index_df(self, domain: str) -> pd.DataFrame:
        idx_key = f"metadata/cleaned/images/index/{domain}/hash_index.parquet"
        data = self._try_get_object(idx_key)
        if data is None or len(data) == 0:
            return pd.DataFrame(columns=["sha256","md5","phash64","object_key_cleaned","item_id","keyword","width","height"])
        return pd.read_parquet(io.BytesIO(data))

    def _write_index_df(self, domain: str, df: pd.DataFrame):
        idx_key = f"metadata/cleaned/images/index/{domain}/hash_index.parquet"
        bio = io.BytesIO()
        df.to_parquet(bio, index=False)
        self._put_bytes(idx_key, bio.getvalue(), content_type="application/octet-stream")

    # ---------- image utils ----------
    def _valid_and_normalize(self, b: bytes) -> tuple[bytes, Image.Image] | tuple[None, None]:
        try:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            w, h = img.size
            if w < self.min_size or h < self.min_size:
                return None, None
            # nén về JPEG (normalize)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=92, optimize=True)
            return out.getvalue(), img
        except Exception:
            return None, None

    @staticmethod
    def _sha256_md5(data: bytes) -> tuple[str, str]:
        return hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest()

    @staticmethod
    def _phash64(img: Image.Image) -> int:
        ph = imagehash.phash(img)          # 8x8 -> 64-bit
        return int(str(ph), 16)            # hex -> int

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        # Python 3.8+ có int.bit_count()
        return (a ^ b).bit_count()

    # ---------- main ----------
    def clean_batch(self, meta_records: list[dict]) -> pd.DataFrame:
        if not meta_records:
            return pd.DataFrame([])

        # gom theo domain để đọc/ghi index 1 lần
        by_domain: dict[str, list[dict]] = {}
        for m in meta_records:
            by_domain.setdefault(m["domain"], []).append(m)

        cleaned_rows = []
        for domain, items in by_domain.items():
            index_df = self._read_index_df(domain)
            # các set/array để tra nhanh
            known_sha = set(index_df["sha256"].tolist()) if len(index_df) else set()
            if len(index_df):
                known_phash = (
                    index_df["phash64"]
                    .dropna()
                    .apply(lambda x: int(x, 16) if isinstance(x, str) else int(x))
                    .tolist()
                )
            else:
                known_phash = []

            accepted_index_rows = []
            duplicate_rows = []

            batch_id = str(uuid.uuid4())
            when = datetime.now(timezone.utc).isoformat()

            for m in items:
                raw_key = m["raw_path"]
                raw_bytes = self._get_bytes(raw_key)
                norm_bytes, pil_img = self._valid_and_normalize(raw_bytes)
                if norm_bytes is None:
                    duplicate_rows.append({
                        **m, "drop_reason": "too_small_or_corrupt", "when": when
                    })
                    continue

                sha256, md5 = self._sha256_md5(norm_bytes)
                if sha256 in known_sha:
                    duplicate_rows.append({
                        **m, "drop_reason": "sha256_dup", "when": when
                    })
                    continue

                # pHash check
                ph64 = self._phash64(pil_img)
                is_dup_phash = False
                for old in known_phash:
                    if self._hamming(ph64, old) <= self.phash_threshold:
                        is_dup_phash = True
                        break
                if is_dup_phash:
                    duplicate_rows.append({
                        **m, "drop_reason": f"phash_dup<= {self.phash_threshold}", "when": when
                    })
                    continue

                # accept -> upload cleaned JPEG
                cleaned_key = raw_key.replace("raw/images/", "cleaned/images/").rsplit(".", 1)[0] + ".jpg"
                self._put_bytes(cleaned_key, norm_bytes, content_type="image/jpeg")

                w, h = pil_img.size
                # row trả về cho pipeline đi tiếp
                out = dict(m)
                out["cleaned_path"] = cleaned_key
                out["width"], out["height"] = int(w), int(h)
                cleaned_rows.append(out)

                # update in-memory index
                known_sha.add(sha256)
                known_phash.append(ph64)
                accepted_index_rows.append({
                    "sha256": sha256,
                    "md5": md5,
                    "phash64": hex(ph64),
                    "object_key_cleaned": cleaned_key,
                    "item_id": m.get("id"),
                    "keyword": m.get("keyword"),
                    "width": int(w),
                    "height": int(h),
                })

            # ghi parquet cleaned metadata + duplicates cho batch
            if cleaned_rows:
                df_clean_batch = pd.DataFrame([r for r in cleaned_rows if r["domain"] == domain])
                if not df_clean_batch.empty:
                    key = f"metadata/cleaned/images/{domain}/{batch_id}.parquet"
                    bio = io.BytesIO()
                    df_clean_batch.to_parquet(bio, index=False)
                    self._put_bytes(key, bio.getvalue(), content_type="application/octet-stream")

            if duplicate_rows:
                df_dup = pd.DataFrame(duplicate_rows)
                key = f"metadata/cleaned/images/{domain}/{batch_id}_duplicates.parquet"
                bio = io.BytesIO()
                df_dup.to_parquet(bio, index=False)
                self._put_bytes(key, bio.getvalue(), content_type="application/octet-stream")

            # cập nhật index
            if accepted_index_rows:
                new_idx = pd.DataFrame(accepted_index_rows)
                index_df = pd.concat([index_df, new_idx], ignore_index=True)
                # loại trùng index (sha256 unique)
                index_df = index_df.drop_duplicates(subset=["sha256"], keep="first")
                self._write_index_df(domain, index_df)

        return pd.DataFrame(cleaned_rows)
