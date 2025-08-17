import os
import numpy as np

from ecg_utils import DEFAULT_DATA_DIR, _resolve_path, load_ecg_data


def test_default_data_dir_exists():
    # 默认目录可以不存在，但路径应可拼接
    assert isinstance(DEFAULT_DATA_DIR, str)


def test_resolve_path_prefers_existing(tmp_path):
    # 当文件存在于当前路径时，_resolve_path 应返回原路径
    p = tmp_path / "foo.dat"
    p.write_text("dummy")
    resolved = _resolve_path(str(p))
    assert resolved == str(p)


def test_load_ecg_data_dicom(tmp_path):
    # 构造一个假的 DICOM 文件路径（本测试仅验证分支路由，不真正读取）
    # 使用 .dcm 后缀应走 DICOM 分支并尝试读取；由于文件为空，函数应返回 (None, None, None)
    fake = tmp_path / "fake.dcm"
    fake.write_bytes(b"")
    signal, fs, lead = load_ecg_data({"filepath": str(fake), "lead_index": "II"})
    assert signal is None and fs is None and lead is None


def test_load_ecg_data_wfdb_missing():
    # 对于不存在的 WFDB 记录，应返回 (None, None, None)
    signal, fs, lead = load_ecg_data({"record_name": "__not_exists__", "lead_index": 0})
    assert signal is None and fs is None and lead is None


