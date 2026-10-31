import os
import sys
import json
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PUBLIC_KEY = 'https://disk.yandex.ru/d/_R6xYg1frCGW0w'
API_RESOURCES_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources'
API_DOWNLOAD_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources/download'

def fetch_resource_metadata(path='', offset=0, limit=100):
    url = f"{API_RESOURCES_URL}?public_key={urllib.parse.quote(PUBLIC_KEY)}&limit={limit}&offset={offset}"
    if path:
        url += f"&path={urllib.parse.quote(path)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))

def get_fresh_download_url(item_path):
    url = f"{API_DOWNLOAD_URL}?public_key={urllib.parse.quote(PUBLIC_KEY)}&path={urllib.parse.quote(item_path)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        return data.get('href')

def crawl_all_files(path=''):
    all_files = []
    offset = 0
    limit = 100
    while True:
        data = fetch_resource_metadata(path, offset=offset, limit=limit)
        embedded = data.get('_embedded', {})
        items = embedded.get('items', [])
        total = embedded.get('total', 0)
        
        for item in items:
            item_rel = (path + '/' + item['name']) if path else ('/' + item['name'])
            if item['type'] == 'dir':
                all_files.extend(crawl_all_files(item_rel))
            else:
                all_files.append({
                    'name': item['name'],
                    'rel_path': item_rel.lstrip('/'),
                    'size': item.get('size', 0),
                    'direct_url': item.get('file')
                })
        
        offset += len(items)
        if offset >= total or not items:
            break
            
    return all_files

def download_file(item, base_dir, max_retries=3):
    rel_path = item['rel_path']
    dest_path = os.path.join(base_dir, rel_path.replace('/', os.sep))
    expected_size = item['size']
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == expected_size:
        return {'status': 'skipped', 'rel_path': rel_path, 'size': expected_size}
    
    download_url = item.get('direct_url')
    
    for attempt in range(1, max_retries + 1):
        try:
            if not download_url or attempt > 1:
                download_url = get_fresh_download_url('/' + rel_path)
            
            temp_path = dest_path + '.part'
            req = urllib.request.Request(download_url, headers={'User-Agent': 'Mozilla/5.0'})
            
            with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, 'wb') as f:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            
            actual_size = os.path.getsize(temp_path)
            if expected_size > 0 and actual_size != expected_size:
                raise IOError(f"Size mismatch: expected {expected_size}, got {actual_size}")
            
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(temp_path, dest_path)
            return {'status': 'downloaded', 'rel_path': rel_path, 'size': actual_size}
            
        except Exception as e:
            if attempt == max_retries:
                return {'status': 'failed', 'rel_path': rel_path, 'error': str(e)}
            time.sleep(attempt * 2)

def main():
    root_meta = fetch_resource_metadata()
    folder_name = root_meta.get('name', 'yandex_disk_download')
    
    # Save into folder_name (e.g. 24.09.20206) inside current working directory
    target_dir = os.path.join(os.getcwd(), folder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Target folder on Yandex Disk: {folder_name}")
    print(f"Local destination: {target_dir}")
    print("Crawling file list...")
    start_crawl = time.time()
    files = crawl_all_files('')
    crawl_time = time.time() - start_crawl
    total_size = sum(f['size'] for f in files)
    print(f"Discovered {len(files)} files totaling {total_size / (1024*1024):.2f} MB ({crawl_time:.1f}s)")
    
    print("Starting parallel download (6 threads)...")
    start_dl = time.time()
    
    downloaded_bytes = 0
    downloaded_count = 0
    skipped_count = 0
    failed_items = []
    
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(download_file, f, target_dir): f for f in files}
        for future in as_completed(futures):
            res = future.result()
            if res['status'] == 'downloaded':
                downloaded_count += 1
                downloaded_bytes += res['size']
                print(f"[{downloaded_count + skipped_count}/{len(files)}] Downloaded: {res['rel_path']} ({res['size'] / (1024*1024):.2f} MB)")
            elif res['status'] == 'skipped':
                skipped_count += 1
                downloaded_bytes += res['size']
                print(f"[{downloaded_count + skipped_count}/{len(files)}] Skipped (already exists): {res['rel_path']}")
            else:
                failed_items.append(res)
                print(f"[{downloaded_count + skipped_count}/{len(files)}] FAILED: {res['rel_path']}: {res.get('error')}")

    total_time = time.time() - start_dl
    print("\n--- Summary ---")
    print(f"Total files: {len(files)}")
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {len(failed_items)}")
    print(f"Total size: {downloaded_bytes / (1024*1024):.2f} MB")
    print(f"Time taken: {total_time:.1f} seconds (Average speed: {(downloaded_bytes / (1024*1024)) / max(total_time, 0.1):.2f} MB/s)")
    
    if failed_items:
        print("\nFailed files:")
        for fi in failed_items:
            print(f"  {fi['rel_path']}: {fi.get('error')}")
        sys.exit(1)

if __name__ == '__main__':
    main()
