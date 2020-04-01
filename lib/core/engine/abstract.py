import time
import weakref
from lib.core.hook.abstract import HookBase

class AbstractTrainer(object):
    def __init__(self, *args, **kwargs):
        self._eval_hooks = []
        self._hooks = []
        raise Exception('NO Implement')

    def set_up(self):
        '''
        In the future, this will set up the kwargs and so on
        '''
        pass

    def _get_time(self):
        '''
        Get the current time
        '''
        return time.strftime('%Y-%m-%d-%H-%M') # 2019-08-07-10-34

    def _register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)
    
    def _register_eval_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.
        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._eval_hooks.extend(hooks)

    def run(self, start_iter, max_iter):
        '''
        The basic loop of implement the algorithm
        '''
        # try:
        self.before_train()
        for i in range(start_iter, max_iter):
            self.before_step(i)
            self.train(i)
            self.after_step(i)
        # except Exception as e:
        #     print('The err in the abstract:', e)
        # finally:
        self.after_train()

    def train(self, current_step):
        '''
        the single step of training the model
        '''
        pass
    
    def before_step(self, current_step):
        '''
        the fucntion before step
        '''
        pass

    def after_step(self, current_step):
        '''
        the function after step
        '''
        pass

    def before_train(self):
        '''
        the fucntion before train function
        '''
        pass
    
    def after_train(self):
        '''
        the function after train fucntion
        '''
        pass

class AbstractInference(object):
    def __init__(self):
        raise Exception('No implement')

    def set_up(self):
        '''
        set up the whole inference
        '''
        pass
    
    def _get_time(self):
        '''
        Get the current time
        '''
        return time.strftime('%Y-%m-%d-%H-%M') # 2019-08-07-10-34
    
    
    def run(self):
        '''
        The basic loop of implement the algorithm
        '''
        self.before_inference()
        self.inference()
        self.after_inference()
    
    def before_inference(self):
        pass

    def inference(self):
        pass

    def after_inference(self):
        pass


