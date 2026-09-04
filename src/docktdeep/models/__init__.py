"""Modelos da disciplina.

`baseline.py` guarda o LightningModule; a arquitetura em si esta repartida em
`blocks`/`cnn`/`embeddings`/`heads`, e os termos contrastivos em `losses`.
"""

from .baseline import Baseline
from .stn import STN

__all__ = ["Baseline", "STN"]
