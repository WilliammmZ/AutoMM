import copy
import torch
import itertools
import logging
from enum import Enum
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union
from fvcore.common.param_scheduler import (
    CosineParamScheduler,
    MultiStepParamScheduler,
    StepWithFixedGammaParamScheduler,
)

from Peach.config import CfgNode

from .lr_scheduler import LRScheduler, WarmupParamScheduler, LRMultiplier

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = copy.deepcopy(cfg)

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.CLIP_VALUE, cfg.NORM_TYPE)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.CLIP_VALUE)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.CLIP_TYPE)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer: Type[torch.optim.Optimizer],
    *,
    per_param_clipper: Optional[_GradientClipper] = None,
    global_clipper: Optional[_GradientClipper] = None,
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """
    assert (
        per_param_clipper is None or global_clipper is None
    ), "Not allowed to use both per-parameter clipping and global clipping"

    def optimizer_wgc_step(self, closure=None):
        if per_param_clipper is not None:
            for group in self.param_groups:
                for p in group["params"]:
                    per_param_clipper(p)
        else:
            # global clipper for future use with detr
            # (https://github.com/facebookresearch/detr/pull/287)
            all_params = itertools.chain(*[g["params"] for g in self.param_groups])
            global_clipper(all_params)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer.__name__ + "WithGradientClip",
        (optimizer,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
    cfg: CfgNode, optimizer: Type[torch.optim.Optimizer]
) -> Type[torch.optim.Optimizer]:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer type to become a new dynamically created class OptimizerWithGradientClip
    that inherits the given optimizer and overrides the `step` method to
    include gradient clipping.

    Args:
        cfg: CfgNode, configuration options
        optimizer: type. A subclass of torch.optim.Optimizer

    Return:
        type: either the input `optimizer` (if gradient clipping is disabled), or
            a subclass of it with gradient clipping included in the `step` method.
    """
    if not cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        return optimizer
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer_type = type(optimizer)
    else:
        assert issubclass(optimizer, torch.optim.Optimizer), optimizer
        optimizer_type = optimizer

    grad_clipper = _create_gradient_clipper(cfg.SOLVER.CLIP_GRADIENTS)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        optimizer_type, per_param_clipper=grad_clipper
    )
    if isinstance(optimizer, torch.optim.Optimizer):
        optimizer.__class__ = OptimizerWithGradientClip  # a bit hacky, not recommended
        return optimizer
    else:
        return OptimizerWithGradientClip
    
    
def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    sgd_args = {
        "params": model.parameters(),
        "lr": cfg.SOLVER.BASE_LR,
        "momentum": cfg.SOLVER.MOMENTUM,
        "nesterov": cfg.SOLVER.NESTEROV,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        "foreach": cfg.SOLVER.FOR_EACH,
    }
    
    return maybe_add_gradient_clipping(cfg, torch.optim.SGD(**sgd_args))



def build_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME

    if name == "WarmupMultiStepLR":
        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
        if len(steps) != len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning(
                "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                "These values will be ignored."
            )
        sched = MultiStepParamScheduler(
            values=[cfg.SOLVER.GAMMA**k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    elif name == "WarmupCosineLR":
        end_value = cfg.SOLVER.BASE_LR_END / cfg.SOLVER.BASE_LR
        assert end_value >= 0.0 and end_value <= 1.0, end_value
        sched = CosineParamScheduler(1, end_value)
    elif name == "WarmupStepWithFixedGammaLR":
        sched = StepWithFixedGammaParamScheduler(
            base_value=1.0,
            gamma=cfg.SOLVER.GAMMA,
            num_decays=cfg.SOLVER.NUM_DECAYS,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    sched = WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
        cfg.SOLVER.RESCALE_INTERVAL,
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)
