import io, os, cv2, numpy as np, pandas as pd, tempfile
from PIL import Image

class VideoPreprocessor:
    def __init__(self, minio_client, bucket: str, frame_rate=1.0, out_size=(224,224), max_frames=64):
        self.client = minio_client
        self.bucket = bucket
        self.frame_rate = frame_rate
        self.out_size = out_size
        self.max_frames = max_frames

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

    def preprocess_one(self, cleaned_key: str) -> dict | None:
        data = self._get_bytes(cleaned_key)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(data)
            path = f.name

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            os.unlink(path); return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        step = int(round(fps / self.frame_rate)) if fps > 0 else 1
        frames = []
        i = 0
        while True and len(frames) < self.max_frames:
            ret = cap.grab()
            if not ret: break
            if i % max(1, step) == 0:
                ok, frame = cap.retrieve()
                if not ok: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img = self._resize_center_crop(img)
                arr = (np.asarray(img, dtype=np.float32) / 255.0)
                frames.append(arr)
            i += 1

        cap.release()
        os.unlink(path)
        if not frames: return None
        stack = np.stack(frames, axis=0).astype("float32")  # [T,H,W,3]

        npy_key = cleaned_key.replace("cleaned/videos/", "preprocessed/videos/").rsplit(".",1)[0] + ".npy"
        bio = io.BytesIO()
        np.save(bio, stack)
        self._put_bytes(npy_key, bio.getvalue())

        return {"tensor_path": npy_key, "num_frames": int(stack.shape[0])}

    def preprocess_batch(self, meta_records: list[dict]) -> pd.DataFrame:
        rows = []
        for m in meta_records:
            out = self.preprocess_one(m["cleaned_path"])
            if out:
                mm = dict(m); mm.update(out)
                rows.append(mm)
        return pd.DataFrame(rows)
