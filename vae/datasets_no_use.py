"""
    根据results中的内容，构建dataloader供模型训练
    每个sample的数据存储在单独的文件夹中，文件夹命名为0, 1, 2, ..., N
    1. 输入：
    每个sample的x为文件夹中sim_params_0.csv的
        mutation_rate	birth_rate	death_rates	aggression	start_time	father	universe
    的内容构成的(m, 7)的矩阵，其中m为该sample的突变数目
    2. label:
    每个sample的y为文件夹中vaf_distribution.csv的normalized列构成的的(100,)的向量
    本文件同时实现将分离的数据打包成一个完整的数据集文件的功能。
"""
from typing import List, Optional, Tuple, Any
import os
import csv
import ast
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:
    Dataset = object  # type: ignore
    torch = None
    _HAS_TORCH = False


def _parse_sim_params(path: str) -> np.ndarray:
    """Parse `sim_params_0.csv` into an (m,7) float array.

    Uses csv.DictReader and ast.literal_eval for fields that may be lists.
    """
    # Try standard CSV parsing first
    expected_cols = [
        'mutation_rate', 'birth_rate', 'death_rates',
        'aggression', 'start_time', 'father', 'universe'
    ]

    def _convert_row_from_mapping(mapping: dict) -> List[float]:
        vals: List[float] = []
        for c in expected_cols:
            if c not in mapping:
                raise KeyError(c)
            v = mapping[c]
            if v is None:
                vals.append(np.nan)
                continue
            v = str(v).strip()
            if v == '':
                vals.append(np.nan)
                continue
            if c == 'death_rates':
                try:
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, (list, tuple)):
                        vals.append(float(np.mean(parsed)))
                    else:
                        vals.append(float(parsed))
                except Exception:
                    vals.append(float(v))
            else:
                vals.append(float(v))
        return vals

    rows: List[List[float]] = []
    # First attempt: csv.DictReader with default settings
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        # If DictReader produced a single-field header (file is whitespace-separated),
        # fall back to manual parsing below
        if reader.fieldnames is not None and len(reader.fieldnames) > 1 and all(col in reader.fieldnames for col in expected_cols):
            for r in reader:
                try:
                    rows.append(_convert_row_from_mapping(r))
                except KeyError:
                    # unexpected: missing column, fall back to manual parse
                    rows = []
                    break

    # If csv.DictReader didn't yield expected columns, parse manually.
    if len(rows) == 0:
        with open(path, 'r', newline='') as f:
            # Read all non-comment, non-empty lines
            lines = [ln.rstrip('\n') for ln in f.readlines()]
        data_lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith('#')]
        if not data_lines:
            return np.zeros((0, 7), dtype=float)
        header_tokens = data_lines[0].strip().split()
        # Build index map for expected columns
        idx_map = {}
        for c in expected_cols:
            if c in header_tokens:
                idx_map[c] = header_tokens.index(c)
            else:
                raise ValueError(f"Column '{c}' not found in {path} (header tokens: {header_tokens})")

        for ln in data_lines[1:]:
            parts = ln.strip().split()
            # If the line is shorter than header, skip it
            if len(parts) < max(idx_map.values()) + 1:
                # skip or pad with empty strings
                # we'll pad to length
                parts = parts + [''] * (max(idx_map.values()) + 1 - len(parts))
            vals: List[float] = []
            for c in expected_cols:
                v = parts[idx_map[c]] if idx_map[c] < len(parts) else ''
                if v == '':
                    vals.append(np.nan)
                    continue
                if c == 'death_rates':
                    try:
                        parsed = ast.literal_eval(v)
                        if isinstance(parsed, (list, tuple)):
                            vals.append(float(np.mean(parsed)))
                        else:
                            vals.append(float(parsed))
                    except Exception:
                        vals.append(float(v))
                else:
                    vals.append(float(v))
            rows.append(vals)

    if len(rows) == 0:
        return np.zeros((0, 7), dtype=float)
    return np.asarray(rows, dtype=float)


def _parse_vaf_distribution(path: str, expect_len: int = 100) -> np.ndarray:
    """Parse `vaf_distribution.csv` and return the `normalized` column as an array

    If the column is shorter than expect_len it will be padded with zeros; if
    longer it will be trimmed.
    """
    vals: List[float] = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if 'normalized' in reader.fieldnames:  # type: ignore
            for r in reader:
                v = r.get('normalized', '')
                if v is None or v == '':
                    vals.append(0.0)
                else:
                    vals.append(float(v))
        else:
            # fallback: try first numeric column
            for r in reader:
                for k in r:
                    v = r[k]
                    try:
                        vals.append(float(v))
                        break
                    except Exception:
                        continue
    arr = np.array(vals, dtype=float)
    if arr.size < expect_len:
        padded = np.zeros((expect_len,), dtype=float)
        padded[: arr.size] = arr
        return padded
    else:
        return arr[:expect_len]


BASE_DATASET = Dataset if _HAS_TORCH else object


class VAFDataset(BASE_DATASET):
    """Dataset that reads samples from a results folder.

    Each sample is a directory named with an integer (0,1,2...). Inside each
    directory we expect `sim_params_0.csv` and `vaf_distribution.csv`.

    __getitem__ returns a tuple (x, y) where:
        - x: numpy array of shape (m,7) (float)
        - y: numpy array of shape (100,) (float)

    If PyTorch is available and `to_torch=True`, tensors are returned instead.
    """

    def __init__(
        self,
        root_dir: str = os.path.join(os.path.dirname(__file__), '..', 'results'),
        sample_list: Optional[List[str]] = None,
        packed_file: Optional[str] = None,
        to_torch: bool = False,
    ) -> None:
        self.to_torch = to_torch and _HAS_TORCH
        self.root_dir = os.path.abspath(root_dir)

        if packed_file is not None:
            # load packed .npz file
            data = np.load(packed_file, allow_pickle=True)
            self.x_list = data['x']
            self.y_list = data['y']
            # x_list may be an object array of arrays
            self.sample_names = data['names'].tolist() if 'names' in data else list(range(len(self.x_list)))
        else:
            # scan directories
            if sample_list is None:
                # directories that are numeric
                names = [n for n in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, n))]
                # filter numeric names and sort
                names = [n for n in names if n.isdigit()]
                names = sorted(names, key=lambda s: int(s))
            else:
                names = sample_list

            self.sample_names = names
            x_acc: List[Any] = []
            y_acc: List[np.ndarray] = []
            for name in names:
                dpath = os.path.join(self.root_dir, name)
                spath = os.path.join(dpath, 'sim_params_0.csv')
                vpath = os.path.join(dpath, 'vaf_distribution.csv')
                if not os.path.exists(spath) or not os.path.exists(vpath):
                    # skip samples missing files
                    continue
                try:
                    x = _parse_sim_params(spath)
                    y = _parse_vaf_distribution(vpath)
                except Exception as e:
                    # skip problematic samples but warn
                    print(f"Warning: failed to parse sample {name}: {e}")
                    continue
                x_acc.append(x)
                y_acc.append(y)

            self.x_list = np.array(x_acc, dtype=object)
            self.y_list = np.stack(y_acc, axis=0) if len(y_acc) > 0 else np.zeros((0, 100), dtype=float)

    def __len__(self) -> int:
        return len(self.y_list)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x = self.x_list[idx]
        y = self.y_list[idx]
        if self.to_torch:
            # perform a local import here so static analyzers won't complain
            # when `torch` is not available at module import time
            import importlib
            _torch = importlib.import_module('torch')
            # Ensure numpy arrays are numeric (not dtype=object) before tensor conversion
            x_np = np.asarray(x)
            if x_np.dtype == object:
                try:
                    x_np = x_np.astype(np.float32)
                except Exception:
                    # sometimes object arrays contain nested lists/arrays; try per-row conversion
                    try:
                        x_np = np.array([np.asarray(r, dtype=np.float32) for r in x_np], dtype=np.float32)
                    except Exception:
                        name = getattr(self, 'sample_names', None)
                        sample_id = name[idx] if name is not None and idx < len(name) else idx
                        raise TypeError(f"Cannot convert sample {sample_id} 'x' to numeric array (object dtype)")

            y_np = np.asarray(y, dtype=np.float32)

            x_t = _torch.tensor(x_np, dtype=_torch.float32)
            y_t = _torch.tensor(y_np, dtype=_torch.float32)
            return x_t, y_t
        return x, y


def pack_results_to_npz(root_dir: str, out_path: str) -> None:
    """Pack all samples under `root_dir` into a single compressed .npz file.

    The .npz will contain:
        - x: object array of per-sample (m,7) arrays
        - y: (N,100) float array
        - names: list of sample folder names
    """
    root_dir = os.path.abspath(root_dir)
    names = [n for n in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, n))]
    names = [n for n in names if n.isdigit()]
    names = sorted(names, key=lambda s: int(s))

    x_acc: List[Any] = []
    y_acc: List[np.ndarray] = []
    used_names: List[str] = []
    for name in names:
        dpath = os.path.join(root_dir, name)
        spath = os.path.join(dpath, 'sim_params_0.csv')
        vpath = os.path.join(dpath, 'vaf_distribution.csv')
        if not os.path.exists(spath) or not os.path.exists(vpath):
            continue
        try:
            x = _parse_sim_params(spath)
            y = _parse_vaf_distribution(vpath)
        except Exception as e:
            print(f"Warning: skipping {name} due to parse error: {e}")
            continue
        x_acc.append(x)
        y_acc.append(y)
        used_names.append(name)

    x_arr = np.array(x_acc, dtype=object)
    y_arr = np.stack(y_acc, axis=0) if len(y_acc) > 0 else np.zeros((0, 100), dtype=float)
    np.savez_compressed(out_path, x=x_arr, y=y_arr, names=np.array(used_names, dtype=object))


if __name__ == '__main__':
    # small demo: build dataset from repo-level results and print shapes
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_data_save_dir = os.path.join(repo_root,"data", "chess", "train")
    results_dir = os.path.join(repo_root, 'results')
    print('Results dir:', results_dir)
    ds = VAFDataset(root_dir=results_dir)
    print('Dataset length:', len(ds))
    for i in range(min(5, len(ds))):
        x, y = ds[i]
        print(f'sample {i}: x.shape={(None if x is None else getattr(x, "shape", None))}, y.shape={y.shape}')

    train_data_file = os.path.join(train_data_save_dir, 'packed_train_data.npz')
    if not os.path.exists(train_data_file):
        pack_results_to_npz(results_dir, train_data_file)
        print('Packed train data saved to:', train_data_file)
    else:
        print('Packed train data already exists at:', train_data_file)

    # test load dataset from packed file
    # train_data_file = '/root/data/wja/project/CHESS.cpp/data/chess/train/packed_train_data.npz'
    # ds = VAFDataset(packed_file=train_data_file, to_torch=True)
    # print('Loaded dataset from packed file:', train_data_file)
    # print('Dataset length:', len(ds))
    # for i in range(min(5, len(ds))):
    #     x, y = ds[i]
    #     print(f'sample {i}: x.shape={(None if x is None else getattr(x, "shape", None))}, y.shape={y.shape}')
