import os

def save_file(save_dir: str, file_name: str, data: bytes):
    filepath = os.path.join(save_dir, file_name)
    with open(filepath, 'wb') as f:
        f.write(data)
        