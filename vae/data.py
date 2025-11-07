"""
- 实际的数据命名为：
> {num_type_number}/{index}/{second_index}/
- 生成方式：
```python
# 均为均匀分布
# mutation rate：0	100
# selective advantage（birth rate）：0	1
# death rate：0	1
# aggression：0	1
# time to new clone：0	15
num_type_number = [1, 2, 4, 8]
index = range(100)
具体参数 = 根据要求的均匀分布采样得到的
second_index = range(100)
```

- 具体执行过程：    
    1. 遍历num_type_number
    2. 遍历index，对每个index，采样num_type_number组参数
    3. 对该index的参数，运行命令`./cancer_gillespie_simulation_no_display [options]`共len(second_index)次，每次的结果存放到对应的second_index文件夹
    4. 每个index文件夹中存放一个csv文件，第一列是index，第二列是具体的该index的执行命令。

- 单次命令示例：
```shell
./cancer_gillespie_simulation_no_display -m 8,12 -b 1.0,1.1 -d 0.05,0.03   -r 0.9,0.95  -t 0,50  -o ./tmp2/
```
"""

import csv
import random
import shlex
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm


# NUM_TYPE_CHOICES = [1, 2, 4, 8]
# NUM_TYPE_CHOICES = [2, 4, 8]
NUM_TYPE_CHOICES = [1, 2]
INDEX_RANGE = range(100)
SECOND_INDEX_RANGE = range(64)
MUTATION_RATE_RANGE = (0.0, 100.0)
SELECTIVE_ADVANTAGE_RANGE = (0.0, 1.0)
DEATH_RATE_RANGE = (0.0, 1.0)
AGGRESSION_RANGE = (0.0, 1.0)
TIME_TO_NEW_CLONE_RANGE = (0.0, 15.0)
BINARY_NAME = 'cancer_gillespie_simulation_no_display'


result_data_dir = Path('/root/data/wja/project/CHESS.cpp/data_original/data')


def _format_values(values):
    """Format parameter values for the CLI, trimming extra zeros."""

    return ','.join(f"{value:.6f}".rstrip('0').rstrip('.') for value in values)


def _sample_parameters(num_types):
    """Sample one parameter set for the requested clone count."""

    def sample(range_limits):
        low, high = range_limits
        return [random.uniform(low, high) for _ in range(num_types)]

    time_to_new_clone = sample(TIME_TO_NEW_CLONE_RANGE)
    if len(time_to_new_clone) > 1:
        time_to_new_clone.sort()

    return {
        'mutation_rates': sample(MUTATION_RATE_RANGE),
        'selective_advantages': sample(SELECTIVE_ADVANTAGE_RANGE),
        'death_rates': sample(DEATH_RATE_RANGE),
        'aggressions': sample(AGGRESSION_RANGE),
        'time_to_new_clone': time_to_new_clone,
    }


def _load_existing_commands(csv_path):
    commands = {}
    if not csv_path.exists():
        return commands

    with csv_path.open('r', newline='') as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                idx = int(row[0])
                second_idx = int(row[1])
            except ValueError:
                continue
            commands.setdefault(idx, {})[second_idx] = row[2]

    return commands


def _write_commands(csv_path, commands_by_index):
    with csv_path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        for idx in sorted(commands_by_index):
            second_map = commands_by_index[idx]
            for second_idx in sorted(second_map):
                writer.writerow([idx, second_idx, second_map[second_idx]])


def _extract_output_dir(command_str):
    parts = shlex.split(command_str)
    for i, token in enumerate(parts):
        if token == '-o' and i + 1 < len(parts):
            return Path(parts[i + 1])
    return None


def _replace_output_dir(command_str, output_dir):
    parts = shlex.split(command_str)
    for i, token in enumerate(parts):
        if token == '-o' and i + 1 < len(parts):
            parts[i + 1] = str(output_dir) + '/'
            break
    return shlex.join(parts)


def _build_command(binary_path, params, output_dir):
    command = [
        str(binary_path),
        '-m', _format_values(params['mutation_rates']),
        '-b', _format_values(params['selective_advantages']),
        '-d', _format_values(params['death_rates']),
        '-r', _format_values(params['aggressions']),
        '-t', _format_values(params['time_to_new_clone']),
        '-o', str(output_dir) + '/',
    ]
    return command


def _run_command(command):
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout

def main():
    binary_path = (Path(__file__).resolve().parents[1] / BINARY_NAME).resolve()
    error_log_path = Path('/root/data/wja/project/CHESS.cpp/data_original/error.log')
    if not binary_path.exists():
        raise FileNotFoundError(f'Binary not found: {binary_path}')

    for num_types in tqdm(NUM_TYPE_CHOICES, desc='num_types'):
        num_root = result_data_dir / str(num_types)
        num_root.mkdir(parents=True, exist_ok=True)

        csv_path = num_root / 'commands.csv'
        commands_by_index = _load_existing_commands(csv_path)

        index_iter = tqdm(INDEX_RANGE, desc=f'index (types={num_types})', leave=False)
        for idx in index_iter:
            index_root = num_root / str(idx)
            index_root.mkdir(parents=True, exist_ok=True)

            index_commands = commands_by_index.get(idx)
            if not index_commands:
                params = _sample_parameters(num_types)
                index_commands = {}
                for second_idx in SECOND_INDEX_RANGE:
                    output_dir = index_root / str(second_idx)
                    command = _build_command(binary_path, params, output_dir)
                    index_commands[second_idx] = shlex.join(command)
                commands_by_index[idx] = index_commands
            else:
                existing_command = next((cmd for cmd in index_commands.values() if cmd), None)
                if existing_command is None:
                    params = _sample_parameters(num_types)
                    index_commands = {}
                    for second_idx in SECOND_INDEX_RANGE:
                        output_dir = index_root / str(second_idx)
                        command = _build_command(binary_path, params, output_dir)
                        index_commands[second_idx] = shlex.join(command)
                    commands_by_index[idx] = index_commands
                else:
                    for second_idx in SECOND_INDEX_RANGE:
                        command_str = index_commands.get(second_idx)
                        if command_str:
                            continue
                        output_dir = index_root / str(second_idx)
                        index_commands[second_idx] = _replace_output_dir(existing_command, output_dir)

            second_iter = tqdm(sorted(index_commands), desc=f'second_idx idx={idx}', leave=False)
            for second_idx in second_iter:
                command_str = index_commands[second_idx]
                output_dir = _extract_output_dir(command_str)
                if output_dir is None:
                    continue

                output_dir.mkdir(parents=True, exist_ok=True)
                marker_file = output_dir / 'vaf_wholetumour_0.csv'
                if marker_file.exists():
                    continue

                if output_dir.exists():
                    for child in output_dir.iterdir():
                        if child.is_dir():
                            shutil.rmtree(child)
                        else:
                            child.unlink()

                rc, output = _run_command(shlex.split(command_str))
                if rc != 0:
                    error_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with error_log_path.open('a', encoding='utf-8') as log_file:
                        log_file.write(
                            f'index={idx}\nsecond_index={second_idx}\nreturn_code={rc}\ncommand={command_str}\noutput={output}\n\n'
                        )

            # _write_commands(csv_path, commands_by_index)

        _write_commands(csv_path, commands_by_index)


if __name__ == '__main__':
    main()


