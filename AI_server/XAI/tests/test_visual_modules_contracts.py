import os
import numpy as np

from explanation_af import generate_af_visuals
from explanation_st import generate_st_visuals
from explanation_lbbb import generate_lbbb_visuals
from explanation_rbbb import generate_rbbb_visuals
from explanation_avb import generate_avb_visuals
from explanation_normal import generate_normal_visuals
from explanation_sb import generate_sb_visuals


def _dummy_signal(fs=250, seconds=2.0):
    t = np.linspace(0, seconds, int(fs*seconds), endpoint=False)
    sig = 0.05*np.sin(2*np.pi*1.0*t)  # very small sine
    return sig, fs, t


def test_modules_return_contract(tmp_path):
    sig, fs, _ = _dummy_signal()
    out_dir = tmp_path

    # AF
    r = generate_af_visuals(sig, fs, "AF test", "II", str(out_dir))
    assert isinstance(r, dict)
    # ST
    r = generate_st_visuals(sig, fs, "ST test", "II", str(out_dir))
    assert isinstance(r, dict)
    # LBBB
    r = generate_lbbb_visuals(sig, fs, "LBBB test", "II", str(out_dir))
    assert isinstance(r, dict) or r is None
    # RBBB (multi-lead, fake record, expect None)
    r = generate_rbbb_visuals("__not_exists__", db_dir=None, output_dir=str(out_dir))
    assert r is None or isinstance(r, dict)
    # AVB1
    r = generate_avb_visuals(sig, fs, "1dAVB test", "II", str(out_dir))
    assert isinstance(r, dict) or r is None
    # NR
    r = generate_normal_visuals(sig, fs, "NR test", "II", str(out_dir))
    assert isinstance(r, dict)
    # SB
    r = generate_sb_visuals(sig, fs, "SB test", "II", str(out_dir))
    assert isinstance(r, dict)


