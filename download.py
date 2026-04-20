import urllib.request
import os
import zipfile
import sys

def download_coco():
    download_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(download_dir, exist_ok=True)
    
    urls = {
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip'
    }
    
    for name, url in urls.items():
        zip_path = os.path.join(download_dir, f"{name}.zip")
        print(f"下载 {name}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print(f"解压 {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        os.remove(zip_path)
        print(f"{name} 完成！")
    
    print(f"全部完成！数据集位于: {os.path.abspath(download_dir)}")

if __name__ == "__main__":
    download_coco()