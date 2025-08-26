import io, numpy as np, pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImagePreprocessor:
    def __init__(self, minio_client, bucket: str, out_size=(224, 224), save_npy: bool=False):
        self.client = minio_client
        self.bucket = bucket
        self.out_size = out_size
        self.save_npy = save_npy

    def _get_bytes(self, key: str) -> bytes:
        resp = self.client.get_object(self.bucket, key)
        try:
            return resp.read()
        finally:
            resp.close(); resp.release_conn()

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

            jpeg_buf = io.BytesIO()
            img.save(jpeg_buf, format="JPEG", quality=90, optimize=True)
            jpeg_key = m["cleaned_path"].replace("cleaned/images/", "preprocessed/images/")\
                                        .rsplit(".",1)[0] + ".jpg"
            self._put_bytes(jpeg_key, jpeg_buf.getvalue(), content_type="image/jpeg")

            out = dict(m)
            out["object_key_processed"] = jpeg_key
            out["width"], out["height"] = img.size

            if self.save_npy:
                arr = (np.asarray(img, dtype=np.float32) / 255.0)  # HWC [0..1]
                npy_buf = io.BytesIO()
                np.save(npy_buf, arr.astype("float32"))
                npy_key = jpeg_key.replace("preprocessed/images/", "preprocessed_npy/images/")\
                                  .rsplit(".",1)[0] + ".npy"
                self._put_bytes(npy_key, npy_buf.getvalue())
                out["tensor_path"] = npy_key

            rows.append(out)

        return pd.DataFrame(rows)
