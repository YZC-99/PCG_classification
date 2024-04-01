import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler,LambdaLR

def get_optimizer(model,name,base_lr):
    if name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr)
    elif name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
    return optimizer


class LambdaStepLR(LambdaLR):

  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    """Use last_epoch for the step counter"""
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v

class PolyLRwithWarmup(_LRScheduler):
    """Linearly warmup learning rate and then linearly decay.

    Args:
        optimizer (:class:`torch.optim.Optimizer`): Wrapped optimizer.
        total_steps (int): Number of total training steps.
        warmup_steps (int, optional): Number of warmup steps, defaults to 0
        last_epoch (int, optional): The index of last epoch, defaults to -1. When last_epoch=-1,
            the schedule is started from the beginning or When last_epoch=-1, sets initial lr as lr.
    """

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0, last_epoch: int = -1, **kwargs):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [(self.last_epoch + 1) / (self.warmup_steps + 1) * lr for lr in self.base_lrs]
        else:
            return [
                (self.total_steps - self.last_epoch) / (self.total_steps - self.warmup_steps) * lr
                for lr in self.base_lrs
            ]
