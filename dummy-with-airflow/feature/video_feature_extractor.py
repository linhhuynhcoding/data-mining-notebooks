import io, numpy as np, torch
import torchvision.models as models
import torchvision.transforms as T

class VideoFeatureExtractor:
    """
    Lấy embedding frame-level bằng ResNet50 rồi average pooling theo thời gian.
    """
    def __init__(self, device="cpu"):
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval().to(device)
        self.device = device
        self.tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def _get_bytes(self, client, bucket, key: str) -> bytes:
        resp = client.get_object(bucket, key)
        b = resp.read(); resp.close(); resp.release_conn()
        return b

    def _put_bytes(self, client, bucket, key: str, b: bytes):
        client.put_object(bucket, key, io.BytesIO(b), length=len(b), content_type="application/octet-stream")

    @torch.no_grad()
    def _embed_frames(self, frames: np.ndarray) -> np.ndarray:
        # frames: [T,H,W,3] in [0..1]
        xs = [self.tf(torch.from_numpy(fr)) for fr in frames]
        x  = torch.stack(xs).to(self.device)   # [T,3,H,W]
        feat = self.model(x).squeeze(-1).squeeze(-1).cpu().numpy()  # [T,2048]
        return feat

    def extract_batch(self, client, bucket, meta_records: list[dict]) -> list[dict]:
        rows = []
        for m in meta_records:
            frames = np.load(io.BytesIO(self._get_bytes(client, bucket, m["tensor_path"])), allow_pickle=False)  # [T,H,W,3]
            feat_t = self._embed_frames(frames)   # [T,2048]
            vec = feat_t.mean(axis=0)             # [2048]
            feat_key = m["tensor_path"].replace("preprocessed/videos/", "features/videos/").rsplit(".",1)[0] + ".npy"
            bio = io.BytesIO(); np.save(bio, vec.astype("float32"))
            self._put_bytes(client, bucket, feat_key, bio.getvalue())
            rows.append({"id": m["id"], "domain": m["domain"], "keyword": m["keyword"],
                         "feature_path": feat_key, "dim": int(vec.shape[0])})
        return rows
