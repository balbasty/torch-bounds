import torch as _torch

from .indexing import *     # noqa: F401, F403
from .padding import *      # noqa: F401, F403
from .types import *        # noqa: F401, F403

from . import indexing      # noqa: F401
from . import padding       # noqa: F401
from . import types         # noqa: F401

__all__ = [
    'indexing',
    'padding',
    'types',
]
__all__ += indexing.__all__
__all__ += padding.__all__
__all__ += types.__all__


if hasattr(getattr(_torch, 'fft', None), 'fft'):

    from .realtransforms import *        # noqa: F401, F403
    from . import realtransforms         # noqa: F401

    __all__ += ['realtransforms']
    __all__ += realtransforms.__all__


from . import _version  # noqa: E402
__version__ = _version.get_versions()['version']
