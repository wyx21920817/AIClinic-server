import numpy as np
from explanation_rbbb import _categorize_qrs


def test_rbbb_qrs_categorization():
    assert _categorize_qrs(0.05) == 'narrow'
    assert _categorize_qrs(0.065) == 'normal'
    assert _categorize_qrs(0.115) == 'borderline'
    assert _categorize_qrs(0.125) == 'wide'


