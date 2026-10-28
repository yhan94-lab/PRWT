import argparse
import csv
import random
import re
from pathlib import Path


DEFAULT_TXT_DIR = Path(r"dataMen\txt")


def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_txt(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(row + "\n")


def get_case_id(sample_name):
    """Convert BraTS-MEN-xxxxx-xxx_slice to BraTS-MEN-xxxxx-xxx."""
    match = re.match(r"^(.+)_\d+$", sample_name)
    return match.group(1) if match else sample_name


def slice_index(sample_name):
    match = re.search(r"_(\d+)$", sample_name)
    return int(match.group(1)) if match else -1


def group_by_case(rows):
    groups = {}
    for row in rows:
        groups.setdefault(get_case_id(row), []).append(row)
    for case_id in groups:
        groups[case_id] = sorted(set(groups[case_id]), key=lambda name: (slice_index(name), name))
    return groups


def flatten_cases(case_ids, case_to_samples):
    rows = []
    for case_id in case_ids:
        rows.extend(case_to_samples[case_id])
    return rows


def make_folds(case_ids, num_folds, seed):
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    folds = [[] for _ in range(num_folds)]
    for index, case_id in enumerate(shuffled):
        folds[index % num_folds].append(case_id)
    return folds


def main():
    parser = argparse.ArgumentParser(
        description="Keep current test set fixed and make K-fold train/val splits from current train+val."
    )
    parser.add_argument("--txt_dir", type=Path, default=DEFAULT_TXT_DIR)
    parser.add_argument("--train_file", default="rtrain.txt")
    parser.add_argument("--val_file", default="rval.txt")
    parser.add_argument("--test_file", default="rtest.txt")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    txt_dir = args.txt_dir
    out_dir = args.out_dir or (txt_dir / f"fold{args.folds}_current_test_seed{args.seed}")

    train_rows = read_txt(txt_dir / args.train_file)
    val_rows = read_txt(txt_dir / args.val_file)
    test_rows = read_txt(txt_dir / args.test_file)

    train_groups = group_by_case(train_rows)
    val_groups = group_by_case(val_rows)
    test_groups = group_by_case(test_rows)

    trainval_groups = {}
    for source in (train_groups, val_groups):
        for case_id, samples in source.items():
            trainval_groups.setdefault(case_id, [])
            trainval_groups[case_id].extend(samples)
    for case_id in trainval_groups:
        trainval_groups[case_id] = sorted(set(trainval_groups[case_id]), key=lambda name: (slice_index(name), name))

    trainval_cases = sorted(trainval_groups)
    test_cases = sorted(test_groups)
    overlap = sorted(set(trainval_cases) & set(test_cases))
    if overlap:
        raise RuntimeError(f"Found {len(overlap)} cases shared by train/val and test, e.g. {overlap[:5]}")

    folds = make_folds(trainval_cases, args.folds, args.seed)

    for fold_index, val_cases in enumerate(folds):
        val_set = set(val_cases)
        train_cases = [case_id for case_id in trainval_cases if case_id not in val_set]
        val_cases = sorted(val_cases)

        fold_dir = out_dir / f"fold{fold_index}"
        fold_train_rows = flatten_cases(train_cases, trainval_groups)
        fold_val_rows = flatten_cases(val_cases, trainval_groups)
        fold_test_rows = flatten_cases(test_cases, test_groups)

        write_txt(fold_dir / args.train_file, fold_train_rows)
        write_txt(fold_dir / args.val_file, fold_val_rows)
        write_txt(fold_dir / args.test_file, fold_test_rows)

if __name__ == "__main__":
    main()
