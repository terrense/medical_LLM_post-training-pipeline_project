"""Thin wrapper to invoke data-juicer's CLI entry point programmatically.

The numpy compat shim below is carried over from rlhf_lab_cloud_kit's own
`scripts_cloud/dj_run.py` (already found to be a real, necessary fix there): newer
numpy removed the deprecated `np.int`/`np.float`/etc. aliases that some of
data-juicer's dependency chain still references, causing an AttributeError before
`process_data.main()` is even reached. Re-registering them here as a compat shim avoids
patching data-juicer's own source.

Usage: python dj_run.py <config.yaml path is passed as --config to data-juicer's own CLI>
"""
import numpy as np

_np_compat = {
    "long": np.int_, "ulong": np.uint, "longlong": np.int64, "ulonglong": np.uint64,
    "int": int, "uint": np.uint, "short": np.int16, "ushort": np.uint16,
    "byte": np.int8, "ubyte": np.uint8, "float": float, "double": np.double,
    "bool": bool, "object": object, "str": str, "unicode": np.str_, "complex": complex,
}
for _name, _val in _np_compat.items():
    if not hasattr(np, _name):
        try:
            setattr(np, _name, _val)
        except Exception:
            pass

import sys
from data_juicer.tools.process_data import main

sys.exit(main())
