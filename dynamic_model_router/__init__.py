"""dynamic_model_router — public package alias for `classifier`.

The package is internally organized as `classifier/...` for historical reasons
(it started as a single-purpose classifier and grew into a router framework).
External users may import either name — the alias re-exports the full public
API verbatim.

    from dynamic_model_router import Router, classify, KeywordPack
    from dynamic_model_router import TaskType, TaskComplexity, ModelTier

Both `from classifier import ...` and `from dynamic_model_router import ...`
return the same objects.
"""

# Re-export every name in classifier.__all__ verbatim. We do `from classifier
# import *` and then mirror __all__ so the two namespaces stay perfectly in
# sync — adding a new export to classifier/__init__.py automatically lights
# it up here.
from classifier import *  # noqa: F401, F403
from classifier import __all__ as _classifier_all
from classifier import __version__

__all__ = list(_classifier_all)
