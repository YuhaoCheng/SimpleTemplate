from .hookcatalog import HookCatalog
from .amc_hooks import get_amc_hooks
from .base import get_base_hooks
from .oc_hooks import get_oc_hooks

from .build_hooks import HookAPI

__all__ = ['HookAPI']

def register_hooks():
    HookCatalog.register('base.VisScoreHook', lambda name:get_base_hooks(name))
    HookCatalog.register('base.TSNEHook', lambda name:get_base_hooks(name))
    HookCatalog.register('amc.AMCEvaluateHook', lambda name:get_amc_hooks(name))
    HookCatalog.register('oc.ClusterHook', lambda name:get_oc_hooks(name))
    HookCatalog.register('oc.OCEvaluateHook', lambda name:get_oc_hooks(name))



register_hooks()
