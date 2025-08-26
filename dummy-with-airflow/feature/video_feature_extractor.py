import io, os, tempfile, numpy as np, torch, pyarrow as pa, pyarrow.parquet as pq, uuid
import cv2
from PIL import Image
import torchvision.models as models

class VideoFeatureExtractor:
    def __init__(self, device="cpu", backbone="resnet18", num_frames=16, shard_size=100):
        self.device = device
        self.num_frames = num_frames
        self.shard_size = shard_size
        if backbone=="resnet50":
            w = models.ResNet50_Weights.DEFAULT; m = models.resnet50(w); self.dim=2048; self.model="resnet50"
        else:
            w = models.ResNet18_Weights.DEFAULT; m = models.resnet18(w); self.dim=512; self.model="resnet18"
        self.backbone = torch.nn.Sequential(*list(m.children())[:-1]).to(device).eval()
        self.mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        self.std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)

    def _get_bytes(self, client,bucket,key): ...
    def _put_file(self, client,bucket,key,local_path): ...

    def _sample_frames_npy(self, arr):
        T=arr.shape[0]; idx=np.linspace(0, T-1, self.num_frames).astype(int)
        return arr[idx].astype("float32")

    def _sample_frames_mp4(self, mp4_bytes):
        fd,tmp= tempfile.mkstemp(suffix=".mp4"); os.write(fd, mp4_bytes); os.close(fd)
        cap=cv2.VideoCapture(tmp); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx=np.linspace(0,total-1,self.num_frames).astype(int); frames=[]
        pos=0; j=0
        while True:
            ok = cap.grab()
            if not ok: break
            if pos==idx[j]:
                ok, fr = cap.retrieve(); 
                if not ok: break
                rgb=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                pil=Image.fromarray(rgb)
                pil=pil.resize((256,256)).crop((16,16,240,240))  # center crop 224
                arr=(np.asarray(pil,dtype=np.float32)/255.0)
                frames.append(arr); j+=1; 
                if j>=len(idx): break
            pos+=1
        cap.release(); os.remove(tmp)
        return None if not frames else np.stack(frames,0).astype("float32")

    def _embed(self, frames):
        x=torch.from_numpy(frames).permute(0,3,1,2).to(self.device)
        x=(x-self.mean)/self.std
        with torch.no_grad():
            feat=self.backbone(x).squeeze(-1).squeeze(-1).cpu().numpy().astype("float32")
        return feat.mean(axis=0)

    def extract_batch(self, client,bucket,meta_rows,out_prefix=None):
        ids=[]; obj_keys=[]; domains=[]; keywords=[]; n_frames=[]; embs=[]
        results=[]
        for m in meta_rows:
            src_key = m.get("tensor_path") or m.get("object_key_processed") or m.get("cleaned_path")
            b = self._get_bytes(client,bucket,src_key)
            frames=None
            if src_key.endswith(".npy"):
                arr=np.load(io.BytesIO(b),allow_pickle=False); frames=self._sample_frames_npy(arr)
            else:
                frames=self._sample_frames_mp4(b)
            if frames is None: continue
            emb=self._embed(frames)
            ids.append(m.get("id")); obj_keys.append(src_key)
            domains.append(m.get("domain")); keywords.append(m.get("keyword"))
            n_frames.append(int(frames.shape[0])); embs.append(emb.tolist())
            results.append({"id": m.get("id"), "dim": self.dim, "model": self.model})
            if out_prefix and len(ids)>=self.shard_size:
                self._flush_parquet(client,bucket,out_prefix,ids,obj_keys,domains,keywords,n_frames,embs)
                ids=[];obj_keys=[];domains=[];keywords=[];n_frames=[];embs=[]
        if out_prefix and ids:
            self._flush_parquet(client,bucket,out_prefix,ids,obj_keys,domains,keywords,n_frames,embs)
        return results

    def _flush_parquet(self, client,bucket,out_prefix, ids,obj_keys,domains,keywords,n_frames,embs):
        tbl=pa.Table.from_pydict({
            "item_id": ids,
            "object_key_processed": obj_keys,
            "domain": domains,
            "keyword": keywords,
            "num_frames_used": n_frames,
            "model": [self.model]*len(ids),
            "dim": pa.array([self.dim]*len(ids), type=pa.int16()),
            "embedding": pa.array(embs, type=pa.list_(pa.float32())),
        })
        tmp= tempfile.mktemp(suffix=".parquet")
        pq.write_table(tbl,tmp,compression="zstd")
        fname=f"part-{uuid.uuid4().hex}.parquet"
        key=f"{out_prefix}/{fname}"
        client.fput_object(bucket,key,tmp)
        os.remove(tmp)
