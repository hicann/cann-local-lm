import os, sys
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

# Monkey-patch BEFORE any torchair import — subsequent imports get patched version
import torchair.core._backend as _backend
_backend.initialize_graph_engine = lambda *a, **kw: None
_backend.finalize_graph_engine = lambda *a, **kw: None

from cann_lm.models.export_epair_with_customop import main
main()
