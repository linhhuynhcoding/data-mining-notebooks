import io, numpy as np, torch
import torchvision.models as models
import torchvision.transforms as T

class ImageFeatureExtractor:
    def __init__(self, device="cpu"):
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])  # remove FC
        self.model.eval().to(device)
        self.device = device
        self.tf = T.Compose([
            T.ToTensor(),  # [0..1], HWC->CHW
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def _get_bytes(self, client, bucket, key: str) -> bytes:
        resp = client.get_object(bucket, key)
        b = resp.read(); resp.close(); resp.release_conn()
        return b

    def _put_bytes(self, client, bucket, key: str, b: bytes):
        client.put_object(bucket, key, io.BytesIO(b), length=len(b), content_type="application/octet-stream")

    @torch.no_grad()
    def _embed_one_tensor(self, arr: np.ndarray) -> np.ndarray:
        # arr: HWC [0..1]
        x = self.tf(torch.from_numpy(arr)).unsqueeze(0).to(self.device)
        feat = self.model(x).squeeze().detach().cpu().numpy()  # 2048
        return feat

    def extract_batch(self, client, bucket, meta_records: list[dict]) -> list[dict]:
        out_rows = []
        for m in meta_records:
            tkey = m["tensor_path"]
            arr = np.load(io.BytesIO(self._get_bytes(client, bucket, tkey)), allow_pickle=False)
            feat = self._embed_one_tensor(arr)  # single image
            feat_key = tkey.replace("preprocessed/images/", "features/images/").rsplit(".",1)[0] + ".npy"
            bio = io.BytesIO(); np.save(bio, feat.astype("float32"))
            self._put_bytes(client, bucket, feat_key, bio.getvalue())

            row = {"id": m["id"], "domain": m["domain"], "keyword": m["keyword"],
                   "feature_path": feat_key, "dim": int(feat.shape[0])}
            out_rows.append(row)
        return out_rows
