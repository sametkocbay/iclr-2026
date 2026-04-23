from __future__ import annotations

from pathlib import Path

from src.data import split_train_val_test


def _make_mock_files(num_groups: int, files_per_group: int) -> list[Path]:
    files: list[Path] = []
    for group_idx in range(num_groups):
        for file_idx in range(files_per_group):
            files.append(Path(f"{1000 + group_idx}_{group_idx}-{file_idx}.npz"))
    return files


def test_split_train_val_test_partitions_groups_without_overlap() -> None:
    files = _make_mock_files(num_groups=20, files_per_group=3)

    train_files, val_files, test_files = split_train_val_test(
        files,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=42,
    )

    assert train_files
    assert val_files
    assert test_files

    train_set = set(train_files)
    val_set = set(val_files)
    test_set = set(test_files)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert train_set.union(val_set).union(test_set) == set(files)


def test_split_train_val_test_small_group_count_returns_same_split() -> None:
    files = _make_mock_files(num_groups=2, files_per_group=2)

    train_files, val_files, test_files = split_train_val_test(
        files,
        val_fraction=0.1,
        test_fraction=0.1,
        seed=0,
    )

    expected = sorted(files)
    assert train_files == expected
    assert val_files == expected
    assert test_files == expected
