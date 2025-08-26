# ... giữ toàn bộ import/hàm sẵn có ...

import io, numpy as np, torch
from torchvision.models import resnet18, ResNet18_Weights
from pyarrow import Table, parquet
import tempfile

class ImageFeatureExtractor:
    """
    Đọc tensor .npy từ preprocessed, trích ResNet18 embedding (512d) và
    ghi Parquet shard {item_id, object_key_processed, embedding, model, dim}.
    """
    def __init__(self, device="cpu", shard_size=2000):
        self.device = device
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        self.backbone = model.to(device).eval()
        self.mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
        self.std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)
        self.shard_size = shard_size
        torch.set_num_threads(1)

    def _get_bytes(self, client, bucket, key): ...
    def _put_file(self, client, bucket, key, local_path): ...

    def _extract_tensor(self, arr):
        x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(self.device)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            feat = self.backbone(x)
        feat = feat.squeeze().cpu().numpy().astype("float32")
        return feat

    def extract_batch(self, client, bucket, meta_rows):
        # meta_rows: list[dict] có "tensor_path", "item_id", "domain", "keyword"
        rows_item_id=[]; rows_obj=[]; rows_emb=[]; rows_model=[]; rows_dim=[]
        results=[]
        for m in meta_rows:
            tkey = m["tensor_path"]
            arr = np.load(io.BytesIO(self._get_bytes(client, bucket, tkey)), allow_pickle=False)
            feat = self._extract_tensor(arr)
            out_key = tkey.replace("preprocessed/images/", "features/images/").rsplit(".",1)[0]+".parquet"
            rows_item_id.append(m["id"]); rows_obj.append(m["tensor_path"]); rows_emb.append(feat.tolist())
            rows_model.append("resnet18"); rows_dim.append(int(feat.shape[0]))
            results.append({"id": m["id"], "feature_dim": int(feat.shape[0]), "feature_model": "resnet18"})
            if len(rows_item_id) >= self.shard_size:
                self._flush_parquet(client, bucket, out_key, rows_item_id, rows_obj, rows_emb, rows_model, rows_dim)
                rows_item_id=[]; rows_obj=[]; rows_emb=[]; rows_model=[]; rows_dim=[]
        if rows_item_id:
            self._flush_parquet(client, bucket, out_key, rows_item_id, rows_obj, rows_emb, rows_model, rows_dim)
        return results

    def _flush_parquet(self, client, bucket, out_prefix, item_ids, obj_keys, embeddings, models, dims):
        tbl = Table.from_pydict({
            "item_id": item_ids,
            "object_key_processed": obj_keys,
            "embedding": embeddings,
            "model": models,
            "dim": dims,
        })
        fd, tmp = tempfile.mkstemp(suffix=".parquet"); os.close(fd)
        parquet.write_table(tbl, tmp, compression="zstd")
        out_key = f"{out_prefix}/part-{uuid.uuid4().hex}.parquet"
        client.fput_object(bucket, out_key, tmp)
        os.remove(tmp)
