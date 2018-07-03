import hashlib
from datetime import datetime
import numpy as np
import subprocess
import numbers

def jsonify_lists_dicts_nparrays(xs):
    if isinstance(xs, list):
        return [jsonify_lists_dicts_nparrays(x) for x in xs]
    elif isinstance(xs, dict):
        return {key: jsonify_lists_dicts_nparrays(x) for key, x in xs.items()}
    elif isinstance(xs, np.ndarray):
        return xs.tolist()
    elif isinstance(xs, numbers.Integral):
        return int(xs)
    elif isinstance(xs, numbers.Real):
        return float(xs)
    else:
        return xs


def _get_timestamp():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def get_git_commit_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()


def git_working_directory_clean():
    dirty = subprocess.call(['git', 'diff-index', '--quiet', 'HEAD', '--'])
    return dirty == 0


def reproducible_hash(arg: str):
    return int(hashlib.md5(arg.encode()).hexdigest(), 16)
