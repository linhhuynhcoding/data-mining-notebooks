import io, numpy as np, pandas as pd
from PIL import Image

class ImagePreprocessor:
    def __init__(self, minio_client, bucket: str, out_size=(224,224)):
        self.client = minio_client
        self.bucket = bucket
        self.out_size = out_size

    def _get_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        return data

    def _put_bytes(self, key: str, b: bytes, content_type="application/octet-stream"):
        self.client.put_object(self.bucket, key, io.BytesIO(b), length=len(b), content_type=content_type)

    def _resize_center_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ow, oh = self.out_size
        scale = max(ow / w, oh / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        left = (nw - ow) // 2
        top  = (nh - oh) // 2
        return img.crop((left, top, left + ow, top + oh))

    def preprocess_batch(self, meta_records: list[dict]) -> pd.DataFrame:
        rows = []
        for m in meta_records:
            b = self._get_bytes(m["cleaned_path"])
            img = Image.open(io.BytesIO(b)).convert("RGB")
            img = self._resize_center_crop(img)
            arr = (np.asarray(img, dtype=np.float32) / 255.0)  # HWC [0..1]
            npy = arr.astype("float32")

            npy_key = m["cleaned_path"].replace("cleaned/images/", "preprocessed/images/")\
                                       .rsplit(".",1)[0] + ".npy"
            bio = io.BytesIO()
            np.save(bio, npy)
            self._put_bytes(npy_key, bio.getvalue())

            out = dict(m)
            out["tensor_path"] = npy_key
            rows.append(out)

        return pd.DataFrame(rows)
