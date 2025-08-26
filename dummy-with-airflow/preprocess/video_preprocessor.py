import io, os, cv2, numpy as np, pandas as pd, tempfile
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VideoPreprocessor:
    def __init__(self, minio_client, bucket: str, frame_rate=1.0, out_size=(224,224), max_frames=64):
        self.client = minio_client
        self.bucket = bucket
        self.frame_rate = frame_rate
        self.out_size = out_size
        self.max_frames = max_frames

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

    def preprocess_one(self, cleaned_key: str) -> dict | None:
        data = self._get_bytes(cleaned_key)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(data)
            in_path = f.name

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            os.unlink(in_path); return None

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 0.0
        step = int(round(fps_src / self.frame_rate)) if fps_src > 0 else 1
        step = max(1, step)

        ow, oh = self.out_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_out:
            out_path = f_out.name
        writer = cv2.VideoWriter(out_path, fourcc, self.frame_rate, (ow, oh))

        num_written = 0
        i = 0
        first_frame_bgr = None

        while num_written < self.max_frames:
            grabbed = cap.grab()
            if not grabbed: break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if not ok: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                pil = self._resize_center_crop(pil)
                rgb_proc = np.asarray(pil)
                bgr_proc = cv2.cvtColor(rgb_proc, cv2.COLOR_RGB2BGR)
                if first_frame_bgr is None:
                    first_frame_bgr = bgr_proc.copy()
                writer.write(bgr_proc)
                num_written += 1
            i += 1

        writer.release()
        cap.release()
        os.unlink(in_path)

        if num_written == 0:
            os.unlink(out_path)
            return None

        # upload mp4
        with open(out_path, "rb") as f:
            mp4_bytes = f.read()
        os.unlink(out_path)

        mp4_key = cleaned_key.replace("cleaned/videos/", "preprocessed/videos/")\
                             .rsplit(".",1)[0] + ".mp4"
        self._put_bytes(mp4_key, mp4_bytes, content_type="video/mp4")

        # thumbnail jpg
        thumb_key = mp4_key.replace("preprocessed/videos/", "preprocessed/videos/thumbs/")\
                           .rsplit(".",1)[0] + ".jpg"
        thumb_buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)).save(
            thumb_buf, format="JPEG", quality=85, optimize=True
        )
        self._put_bytes(thumb_key, thumb_buf.getvalue(), content_type="image/jpeg")

        return {
            "object_key_processed": mp4_key,
            "thumbnail_key": thumb_key,
            "num_frames": int(num_written),
            "width": ow,
            "height": oh,
            "target_fps": float(self.frame_rate),
        }

    def preprocess_batch(self, meta_records: list[dict]) -> pd.DataFrame:
        rows = []
        for m in meta_records:
            out = self.preprocess_one(m["cleaned_path"])
            if out:
                mm = dict(m); mm.update(out)
                rows.append(mm)
        return pd.DataFrame(rows)
