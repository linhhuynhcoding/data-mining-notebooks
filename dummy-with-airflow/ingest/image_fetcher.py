import requests
import os
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv

# Cấu hình
API_URL = 'https://api.pexels.com/v1/search'  # <- Đổi thành URL API thật
DOMAINS = ["River in Mountain Forest", "Green Tree Forest", "sea water"]
SAVE_DIR = '../coding/data-mining-notebooks/data/raw/image'
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Tạo thư mục nếu chưa có
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Hàm tải ảnh
def download_image(url: str, save_dir: str):
    try:
        os.makedirs(save_dir, exist_ok=True)

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"✅ Đã tải: {filepath}")
    except Exception as e:
        print(f"❌ Lỗi với {url}: {e}")

def api_fetch_images():
    for i in DOMAINS:
        # Gọi API
        try:
            response = requests.get(API_URL + "?query=" + i +"&per_page=10", timeout=10, headers={"Authorization": API_KEY})
            response.raise_for_status()
            data = response.json()  # Giả sử API trả về danh sách URL ảnh
        except Exception as e:
            print(f"Lỗi khi gọi API: {e}")
            exit()
            
        photos = data.get("photos", [])
        print(f"🔍 Tìm thấy {len(photos)} ảnh với query")

        for photo in photos:
            img_url = photo.get("src", {})
            if img_url:
                download_image(img_url.get("original", {}), SAVE_DIR + "/" + i.replace(" ", "_"))
            else:
                print(f"⚠️ Bỏ qua ảnh không cùng domain hoặc thiếu URL: {img_url}")
