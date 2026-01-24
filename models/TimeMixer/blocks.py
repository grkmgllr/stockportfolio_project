from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn

ActivationName = Literal["gelu"]

def makeActivation(name: ActivationName) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")

class TemporalLinearMixer(nn.Module):
    #TODO: implement class


class MultiscaleSeasonMixing(nn.Module):
    #TODO: implement class


class MultiscaleTrendMixing(nn.Module):
    #TODO: implement class