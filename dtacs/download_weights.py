import requests
import shutil
from tqdm import tqdm


def download_weights(filename:str, link_weights:str, display_progress_bar:bool=True) -> str:
    filename_tmp = filename+".tmp"

    with requests.get(link_weights, stream=True, allow_redirects=True) as r_link:
        total_size_in_bytes = int(r_link.headers.get('content-length', 0))
        r_link.raise_for_status()
        block_size = 8192  # 1 Kibibyte
        with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, disable=not display_progress_bar) as progress_bar:
            with open(filename_tmp, 'wb') as f:
                for chunk in r_link.iter_content(chunk_size=block_size):
                    if display_progress_bar:
                        progress_bar.update(len(chunk))
                    f.write(chunk)

    shutil.move(filename_tmp, filename)

    return filename