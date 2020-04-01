from .hookcatalog import HookCatalog

class HookAPI(object):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.hook_names = cfg.MODEL.hooks
        # self.eval_hook_names = cfg.MODEL.eval_hooks
        self.logger = logger
    def __call__(self):
        self.logger.info(f'*********use hooks:{self.hook_names}')
        hooks = []
        for name in self.hook_names:
            # prefix = name.split('.')[0]
            hook_name = name.split('.')[1]
            temp = HookCatalog.get(name, hook_name)
            hooks.append(temp)
        self.logger.info(f'build:{hooks}')

        # self.logger.info(f'*********use eval_hooks:{self.hook_names}')
        # eval_hooks = []
        # for name in self.eval_hook_names:
        #     # prefix = name.split('.')[0]
        #     hook_name = name.split('.')[1]
        #     temp = HookCatalog.get(name, hook_name)
        #     eval_hooks.append(temp)
        # self.logger.info(f'build:{hooks}')

        # return eval_hooks, hooks
        return hooks

