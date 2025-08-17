import os
import pandas as pd
import imagehash
from PIL import Image

SAVE_DIR = '../coding/data-mining-notebooks/data/cleaned/image'
DATA_DIR = "../coding/data-mining-notebooks/data/metadata"

def clean(id: str, file_path: str, min_width: int, min_height: int):
    file_name = os.path.basename(file_path)
    try:
        img = Image.open(file_path)
        img.verify()
        img = Image.open(file_path).convert("RGB")
        
        img_format = img.format
    except:
        print("Ảnh không hợp lệ")
        return {"": id, "ok": False, "reason": "invalid image"}

    if img_format not in {"JPEG", "PNG"}:
        return {"key": id, "ok": False, "reason": f"wrong format: {img_format}"}

    w, h = img.size
    if w < min_width or h < min_height:
        return {"key": id, "ok": False, "reason": "too small"}

    ph = str(imagehash.phash(img))

    output_path = file_path.replace("raw", "cleaned")
    if output_path:
        img.save(output_path, format="JPEG", quality=92)

    return {
        "key": id,
        "ok": True,
        "phash": ph,
        "output_path": output_path
    }

def clean_image(**ctx):
    results = ctx['ti'].xcom_pull(key="results", task_ids='extract.fetch_api') or []
    print("results", results)
    for res in results:
        cleaned_images = []
        batch_id = res["batch_id"]
        images = res["images"]
        
        for image in images:
            clean_result = clean(image["id"], image["local_path"], 64, 64)
            if (clean_result["ok"]):
                image["clean_path"] = clean_result["output_path"]
                cleaned_images.append(image)
        
        file_path = f"{DATA_DIR}/{res["domain"]}/{res["keyword"]}/{batch_id}_cleaned.parquet"
        pd.DataFrame(cleaned_images).to_parquet(file_path)        
    
