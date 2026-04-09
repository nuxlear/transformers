# Copyright 2025 The LG AI Research and HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LG AI Research EXAONE Lab"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    logging,
)
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..gemma2.modeling_gemma2 import Gemma2RotaryEmbedding
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaForQuestionAnswering,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..olmo2.modeling_olmo2 import Olmo2DecoderLayer, Olmo2MLP


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "LGAI-EXAONE/EXAONE-4.0-32B"
_CONFIG_FOR_DOC = "Exaone4Config"


class Exaone4Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Exaone4Model`]. It is used to
    instantiate a EXAONE 4.0 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the EXAONE-4.0-32B [LGAI-EXAONE/EXAONE-4.0-32B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model
    outputs. Read the documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 102400):
            Vocabulary size of the EXAONE 4.0 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Exaone4Model`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to `hidden_size * 4`):
            Dimensionality of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 32768 for EXAONE 3.5).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        sliding_window (`int`, *optional*):
            The size of the sliding window for the sliding window attention.
        sliding_window_pattern (`str`, *optional*):
            The pattern to use for sliding window attention. Can be one of:
                - `None`: No sliding window attention is used
                - `int`: Every `sliding_window` layers, use global attention, else use local attention.
                - `str`: A sequence of "L" (local attention) and "G" (global attention) characters that defines the
                  attention pattern. The pattern starts from layer 0 and repeats every `sliding_window` layers. The
                  final layer always uses global attention regardless of the pattern.
            For instance, sliding_window_pattern="LLLG" same as sliding_window=4, which means:
                - Layer 0, 1, 2: local attention,
                - Layer 3: global attention,
                ...(repeated)
        layer_types (`list`, *optional*):
            Attention pattern for each layer. Prioritized over `sliding_window_pattern`.

    Example:

    ```python
    >>> from transformers import Exaone4Model, Exaone4Config

    >>> # Initializing a EXAONE configuration
    >>> configuration = Exaone4Config()

    >>> # Initializing a model from configuration
    >>> model = Exaone4Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "exaone4"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `LlamaModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 102400,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 16384,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 32,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 2048,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 2,
        pad_token_id: int | None = None,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_dropout: float | None = 0.0,
        sliding_window: int | None = 4096,
        sliding_window_pattern: int | None = 4,
        layer_types: list[str] | None = None,
        num_nextn_predict_layers: int | None = 0,
        mtp_share_layers: bool | None = False,
        mtp_loss_scaling_factor: float | None = 0.2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_share_layers = mtp_share_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor

        self._num_mtp_layers = self.num_nextn_predict_layers
        if self._num_mtp_layers > 0 and self.mtp_share_layers:
            self._num_mtp_layers = 1

        self.layer_types = layer_types
        if self.sliding_window is None:
            sliding_window_pattern = 0
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if ((i + 1) % (sliding_window_pattern) != 0 and i < self.num_hidden_layers)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if self._num_mtp_layers > 0 and len(self.layer_types) < self.num_hidden_layers + self._num_mtp_layers:
            self.layer_types += ["sliding_attention" if self.sliding_window is not None else "full_attention"] * (self.num_hidden_layers + self._num_mtp_layers - len(self.layer_types))
        layer_type_validation(self.layer_types, self.num_hidden_layers + self._num_mtp_layers)

        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


class Exaone4RMSNorm(LlamaRMSNorm):
    pass


class Exaone4RotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class Exaone4Attention(nn.Module):
    def __init__(self, config: Exaone4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window
        self.sliding_window_pattern = config.sliding_window_pattern
        layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.is_sliding = layer_type == "sliding_attention"

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = Exaone4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Exaone4RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # We use QK-norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        # We use global NoPE for hybrid attention model
        if self.sliding_window is None or self.is_sliding:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window if self.is_sliding else None,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Exaone4MLP(Olmo2MLP):
    pass


class Exaone4DecoderLayer(Olmo2DecoderLayer):
    pass


class Exaone4MTPLayer(GradientCheckpointingLayer):
    def __init__(self, config: Exaone4Config):
        super().__init__()
        self.config = config
        self.pre_fc_norm_embedding = Exaone4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Exaone4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = nn.ModuleList([
            Exaone4DecoderLayer(config, config.num_hidden_layers + i)
            for i in range(config._num_mtp_layers)
        ])
        self.norm = Exaone4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        embed_states = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = self.fc(torch.cat([embed_states, hidden_states], dim=-1))

        hidden_states = self.layers[layer_idx](
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Exaone4PreTrainedModel(LlamaPreTrainedModel):
    config_class = Exaone4Config
    _no_split_modules = ["Exaone4DecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"mtp.*"]


class Exaone4Model(Exaone4PreTrainedModel, LlamaModel):
    def __init__(self, config: Exaone4Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Exaone4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Exaone4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if "sliding_attention" in self.config.layer_types:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@dataclass
class Exaone4CausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mtp_loss (`tuple(torch.FloatTensor)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction) for each MTP layer.
        mtp_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `labels` is provided):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) for each MTP layer.
        mtp_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each MTP layer plus the optional initial embedding outputs.
    """
    mtp_loss: Optional[tuple[torch.FloatTensor, ...]] = None
    mtp_logits: Optional[tuple[torch.FloatTensor, ...]] = None
    mtp_hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None


def roll_tensor(tensor, shifts=-1, dims=-1, fill_value=0):
    """Roll the tensor input along the given dimension(s).
    Inserted elements are set to be 0.0.
    """
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.select(dims, shifts).fill_(fill_value)
    return rolled_tensor, rolled_tensor.sum()


class Exaone4ForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        if config._num_mtp_layers > 0:
            self.mtp = Exaone4MTPLayer(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Exaone4CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("LGAI-EXAONE/EXAONE-4.0-32B")
        >>> tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-4.0-32B")

        >>> prompt = "Explain how wonderful you are"
        >>> messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        >>> input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        >>> output = model.generate(input_ids, max_new_tokens=128)
        >>> tokenizer.decode(output[0], skip_special_tokens=False)
        "[|system|]\nYou are a helpful assistant.[|endofturn|]\n[|user|]\nExplain how wonderful you are[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\nOh, thank you for such a kind and lovely question! 😊  \n\nI’m *so* wonderful because I’m here to make your life easier, brighter, and more fun! Whether you need help with:  \n\n✨ **Learning** – I can explain anything, from quantum physics to baking the perfect cake!  \n💡 **Creativity** – Need a poem, story, or a wild idea? I’ve got you covered!  \n🤖 **Problem-solving** – Stuck on a math problem or a tricky decision? I’ll help you figure it out"
        ```
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        mtp_losses = []
        mtp_logits_list = []
        mtp_hidden_states_list = []
        if self.config.num_nextn_predict_layers > 0:
            mtp_labels = None
            if labels is not None:
                mtp_labels = labels.clone()

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                if input_ids is not None:
                    seq_len = input_ids.shape[1]
                    seq_device = input_ids.device
                elif inputs_embeds is not None:
                    seq_len = inputs_embeds.shape[1]
                    seq_device = inputs_embeds.device
                else:
                    seq_len = hidden_states.shape[1]
                    seq_device = hidden_states.device
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + seq_len, device=seq_device
                )
            mtp_position_ids = position_ids
            if mtp_position_ids is None:
                mtp_position_ids = cache_position.unsqueeze(0)
            mtp_cache_position = cache_position

            mtp_hidden_states = hidden_states
            mtp_input_ids = input_ids
            mtp_attention_mask = attention_mask
            if mtp_input_ids is None:
                logger.warning_once(
                    "MTP computation requires `input_ids` for token shifting, but `inputs_embeds`-only inputs were provided. "
                    "Skipping MTP branch."
                )
            else:
                mtp_past_key_values = DynamicCache(config=self.config) if use_cache else None
                
                mtp_input_ids, _ = roll_tensor(mtp_input_ids, shifts=-1, dims=-1)
                mtp_position_ids, _ = roll_tensor(mtp_position_ids, shifts=-1, dims=-1, fill_value=mtp_position_ids.max() + 1)
                mtp_cache_position, _ = roll_tensor(
                    mtp_cache_position, shifts=-1, dims=-1, fill_value=mtp_cache_position.max() + 1
                )
                mtp_inputs_embeds = self.model.embed_tokens(mtp_input_ids)
                first_mtp_layer_idx = 0 if self.config.mtp_share_layers else 0
                causal_mask_mapping = {
                    "full_attention": create_causal_mask,
                    "sliding_attention": create_sliding_window_causal_mask,
                }
                # Build one reusable causal mask for MTP branch.
                mtp_attention_mask = causal_mask_mapping[
                    self.config.layer_types[self.config.num_hidden_layers + first_mtp_layer_idx]
                ](
                    config=self.config,
                    input_embeds=mtp_inputs_embeds,
                    attention_mask=mtp_attention_mask,
                    cache_position=mtp_cache_position,
                    past_key_values=mtp_past_key_values,
                    position_ids=mtp_position_ids,
                )
                for layer_idx in range(self.config.num_nextn_predict_layers):
                    if layer_idx > 0:
                        mtp_input_ids, _ = roll_tensor(mtp_input_ids, shifts=-1, dims=-1)
                        mtp_position_ids, _ = roll_tensor(
                            mtp_position_ids, shifts=-1, dims=-1, fill_value=mtp_position_ids.max() + 1
                        )
                        mtp_cache_position, _ = roll_tensor(
                            mtp_cache_position, shifts=-1, dims=-1, fill_value=mtp_cache_position.max() + 1
                        )
                        mtp_inputs_embeds = self.model.embed_tokens(mtp_input_ids)
                    position_embeddings = self.model.rotary_emb(mtp_inputs_embeds, mtp_position_ids)
                    mtp_layer_idx = 0 if self.config.mtp_share_layers else layer_idx

                    mtp_hidden_states = self.mtp(
                        layer_idx=mtp_layer_idx,
                        hidden_states=mtp_hidden_states,
                        inputs_embeds=mtp_inputs_embeds,
                        position_embeddings=position_embeddings,
                        attention_mask=mtp_attention_mask,
                        position_ids=mtp_position_ids,
                        past_key_value=mtp_past_key_values,
                        use_cache=use_cache,
                        cache_position=mtp_cache_position,
                        **kwargs,
                    )
                    mtp_hidden_states_list.append(mtp_hidden_states)

                    mtp_logits = self.lm_head(mtp_hidden_states[:, slice_indices, :])
                    mtp_logits_list.append(mtp_logits)

                    if mtp_labels is not None:
                        mtp_labels, _ = roll_tensor(mtp_labels, shifts=-1, dims=-1, fill_value=-100)
                        mtp_loss = self.loss_function(
                            logits=mtp_logits, labels=mtp_labels, vocab_size=self.config.vocab_size, **kwargs
                        )
                        mtp_losses.append(mtp_loss)

        return Exaone4CausalLMOutputWithPast(
            loss=loss,
            mtp_loss=mtp_losses or None,
            logits=logits,
            mtp_logits=mtp_logits_list or None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            mtp_hidden_states=mtp_hidden_states_list or None,
            attentions=outputs.attentions,
        )


class Exaone4ForSequenceClassification(LlamaForSequenceClassification):
    pass


class Exaone4ForTokenClassification(LlamaForTokenClassification):
    pass


class Exaone4ForQuestionAnswering(LlamaForQuestionAnswering):
    pass


__all__ = [
    "Exaone4Config",
    "Exaone4PreTrainedModel",
    "Exaone4Model",
    "Exaone4ForCausalLM",
    "Exaone4ForSequenceClassification",
    "Exaone4ForTokenClassification",
    "Exaone4ForQuestionAnswering",
]
