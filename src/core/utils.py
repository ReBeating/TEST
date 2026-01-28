import os
import json
from collections import OrderedDict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle as pkl

def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        content = pkl.load(f)
        return content

def write_pickle(content, file_path):
    dir_path = os.path.dirname(file_path)
    if not path_exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'wb') as f:
        pkl.dump(content, f)

def read_text(file_path, **kwargs):
    with open(file_path, 'r', **kwargs) as f:
        content = f.read()
        return content
    
def write_text(content, file_path):
    dir_path = os.path.dirname(file_path)
    if not path_exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'w') as f:
        f.write(content)
        
def path_exists(path):
    if os.path.exists(path) or path == '' or path == '.' or path == './':
        return True
    else:
        return False
    
def read_json(file_path, **kwargs):
    file_path = os.path.abspath(file_path)
    with open(file_path, 'rt', **kwargs) as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, file_path, **kwargs):
    dir_path = os.path.dirname(file_path)
    if not path_exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.abspath(file_path)
    with open(file_path, 'wt', **kwargs) as handle:
        json.dump(content, handle, indent=4)

def multiprocessing_wrapper(target_func, args_list, nproc=32,
                            ordered=True, show_progress=True, collect_results=True, desc="Processing"):
    results = [] if collect_results else None
    total = len(args_list)

    with ProcessPoolExecutor(max_workers=nproc) as executor:
        futures = [executor.submit(target_func, *args) for args in args_list]

        if show_progress:
            progress_iter = tqdm(as_completed(futures), total=total, desc=desc)
        else:
            progress_iter = as_completed(futures)

        if collect_results:
            if ordered:
                results = [None] * total
                for i, f in enumerate(futures):
                    try:
                        results[i] = f.result()
                    except Exception as e:
                        results[i] = None
            else:
                results = []
                for f in progress_iter:
                    try:
                        results.append(f.result())
                    except Exception as e:
                        results.append(None)
        else:
            for _ in progress_iter:
                pass

    return results