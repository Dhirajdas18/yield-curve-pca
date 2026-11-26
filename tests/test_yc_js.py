import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from yc import pca

JS = Path(__file__).parents[1] / "src" / "yc" / "yc.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_js_pca_matches_python_signs_included():
    rng = np.random.default_rng(2)
    n = 10
    mat = np.arange(1, n + 1)
    level, slope = np.ones(n), (mat - mat.mean()) / 5
    curve = (mat - 5.5) ** 2 / 20 - 0.4
    X = (rng.normal(0, 30, (200, 1)) * level + rng.normal(0, 10, (200, 1)) * slope
         + rng.normal(0, 5, (200, 1)) * curve + rng.normal(0, 2, (200, n)))
    code = (f"const YC = require({json.dumps(str(JS))}); const X = {json.dumps(X.tolist())};"
            "console.log(JSON.stringify(YC.pca(X)));")
    js = json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True,
                                   check=True).stdout)
    py = pca.pca(X)
    assert np.allclose(js["values"], py["values"], rtol=1e-8)
    assert np.allclose(np.array(js["loadings"])[:, :3], py["loadings"][:, :3], atol=1e-8)
    assert np.allclose(np.array(js["scores"])[:, :3], py["scores"][:, :3], atol=1e-6)
    L = py["loadings"]
    assert L[:, 0].sum() > 0 and L[-1, 1] > L[0, 1] and L[5, 2] < (L[0, 2] + L[-1, 2]) / 2
    assert py["share"][0] > 0.8
