# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_qdelta import QDeltaConfig
from .modeling_qdelta import QDeltaForCausalLM, QDeltaModel

AutoConfig.register("qdelta", QDeltaConfig, True)
AutoModel.register(QDeltaConfig, QDeltaModel, True)
AutoModelForCausalLM.register(QDeltaConfig, QDeltaForCausalLM, True)

__all__ = ['QDeltaConfig', 'QDeltaForCausalLM', 'QDeltaModel']
