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


import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.exaone4.configuration_exaone4 import Exaone4Config
from transformers.models.exaone4.modeling_exaone4 import (
    Exaone4CausalLMOutputWithPast,
    Exaone4Model,
    Exaone4MTPLayer,
    Exaone4PreTrainedModel,
    Exaone4RMSNorm,
    roll_tensor,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMLP,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
)
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import (
    Qwen2_5_VLProcessor,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import (
    Qwen2VLImageProcessorFast,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    PatchEmbed,
    PatchMerger,
    VisionRotaryEmbedding,
)
from transformers.models.qwen2_vl.video_processing_qwen2_vl import (
    Qwen2VLVideoProcessor,
)

from ... import initialization as init
from ...configuration_utils import PretrainedConfig
from ...masking_utils import create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import (
    TransformersKwargs,
    is_grouped_mm_available,
    logging,
)


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "LGAI-EXAONE/EXAONE-4.5-Beta"
_CONFIG_FOR_DOC = "Exaone4_5_Config"


class Exaone4_5_VisionConfig(Qwen2_5_VLVisionConfig):
    model_type = "exaone4_5_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        hidden_size=3584,
        hidden_act="silu",
        intermediate_size=3420,
        num_heads=16,
        in_channels=3,
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        tokens_per_second=4,
        window_size=112,
        out_hidden_size=3584,
        fullatt_block_indexes=[7, 15, 23, 31],
        initializer_range=0.02,
        num_key_value_heads=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_key_value_heads = num_key_value_heads


class Exaone4_5_TextConfig(Exaone4Config):
    model_type = "exaone4_5_text"
    base_config_key = "text_config"
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
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)


class Exaone4_5_Config(PretrainedConfig):
    """
    Exaone 4.5 config
    """

    model_type = "exaone4_5"
    sub_configs = {"vision_config": Exaone4_5_VisionConfig, "text_config": Exaone4_5_TextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=67,
        video_token_id=68,
        **kwargs,
    ):
        # We need to init super() here so that it does not reset values
        # that are in text config to the BaseClass defaults. The Base
        # config has many text related defaults and not all defaults are same as for `Exaone4_5_TextConfig`
        super().__init__(**kwargs)

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        # Attention implementation to use. It sets it recursively on sub-configs so we call it again in the end
        self._attn_implementation = kwargs.pop("attn_implementation", None)

    def __setattr__(self, key, value):
        if (
            (text_config := super().__getattribute__("__dict__").get("text_config")) is not None
            and key not in ["dtype", "architectures", "_attn_implementation_internal", "model_type"]
            and key in text_config.__dict__
        ):
            setattr(text_config, key, value)
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, key):
        if "text_config" in super().__getattribute__("__dict__") and key not in [
            "dtype",
            "architectures",
            "_attn_implementation_internal",
            "model_type",
        ]:
            text_config = super().__getattribute__("text_config")
            if key in text_config.__dict__:
                return getattr(text_config, key)

        return super().__getattribute__(key)


class Exaone4_5_RMSNorm(Exaone4RMSNorm):
    pass


class Exaone4_5_PatchEmbed(PatchEmbed):
    pass


class Exaone4_5_VisionRotaryEmbedding(VisionRotaryEmbedding):
    pass


class Exaone4_5_PatchMerger(PatchMerger):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__(dim, context_dim, spatial_merge_size)
        self.ln_q = Exaone4_5_RMSNorm(context_dim, eps=1e-6)


class Exaone4_5_VisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, config: Exaone4_5_VisionConfig):
        super().__init__(config)
        del self.qkv
        del self.num_key_value_groups

        self.num_key_value_groups = config.num_key_value_heads
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_key_value_groups * self.head_dim

        if self.num_key_value_groups == 1:
            self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        else:
            self.qkv = nn.Linear(self.dim, self.q_dim + (self.kv_dim * 2), bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        if self.num_key_value_groups == 1 :
            query_states, key_states, value_states = (
                self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            )
        else :
            qkv = self.qkv(hidden_states)
            q, kv = torch.split(qkv, [self.q_dim, 2 * self.kv_dim], dim=-1)

            # q: [seq, num_heads, head_dim]
            query_states = q.view(seq_length, self.num_heads, self.head_dim)

            # kv: [seq, 2, num_key_value_groups, head_dim]
            kv = kv.view(seq_length, 2, self.num_key_value_groups, self.head_dim)
            key_states = kv[:, 0]    # [seq, num_key_value_groups, head_dim]
            value_states = kv[:, 1]  # [seq, num_key_value_groups, head_dim]

            # repeat kv
            repeat_factor = self.num_heads // self.num_key_value_groups
            key_states = key_states.repeat_interleave(repeat_factor, dim=1)
            value_states = value_states.repeat_interleave(repeat_factor, dim=1)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if is_flash_attention_requested(self.config):
            # Flash Attention 2: Use cu_seqlens for variable length attention
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            # Other implementations: Process each chunk separately
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]

            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class Exaone4_5_MLP(Qwen2_5_VLMLP):
    pass


class Exaone4_5_VisionBlock(Qwen2_5_VLVisionBlock):
    def __init__(self, config: Exaone4_5_VisionConfig):
        super().__init__(config)
        self.norm1 = Exaone4_5_RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Exaone4_5_RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Exaone4_5_VisionAttention(config)
        self.mlp = Exaone4_5_MLP(config, bias=True)


class Exaone4_5_MTPLayer(Exaone4MTPLayer):
    pass


class Exaone4_5_CausalLMOutputWithPast(Exaone4CausalLMOutputWithPast):
    pass


class Exaone4_5_PreTrainedModel(Exaone4PreTrainedModel):
    config: Exaone4_5_Config
    _no_split_modules = ["Exaone4_5_VisionBlock", "Exaone4_5_DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    _can_compile_fullgraph = (
        is_grouped_mm_available()
    )  # https://huggingface.co/docs/transformers/experts_interface#torchcompile
    _keys_to_ignore_on_load_unexpected = [r"mtp.*"]

    def _init_weights(self, module):
        PreTrainedModel._init_weights(module)
        if isinstance(module, Exaone4_5_VisionRotaryEmbedding):
            inv_freq = 1.0 / (module.theta ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim))
            init.copy_(module.inv_freq, inv_freq)


class Exaone4_5_VisionPreTrainedModel(Exaone4_5_PreTrainedModel, Qwen2_5_VisionTransformerPretrainedModel):
    config_class = Exaone4_5_VisionConfig
    _no_split_modules = ["Exaone4_5_VisionBlock"]

    def __init__(self, config: Exaone4_5_VisionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.patch_embed = Exaone4_5_PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Exaone4_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Exaone4_5_VisionBlock(config) for _ in range(config.depth)])
        self.merger = Exaone4_5_PatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False


class Exaone4_5_TextModel(Exaone4_5_PreTrainedModel, Exaone4Model):
    config_class = Exaone4_5_TextConfig


class Exaone4_5_Model(Exaone4_5_PreTrainedModel, Qwen2_5_VLModel):
    config_class = Exaone4_5_Config
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}

    def __init__(self, config: Exaone4_5_Config):
        super().__init__(config)
        self.visual = Exaone4_5_VisionPreTrainedModel._from_config(config.vision_config)
        self.language_model = Exaone4_5_TextModel._from_config(config.text_config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        r"""
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Exaone4_5_ImageProcessor.__call__`] for details. [`Exaone4_5_Processor`] uses
            [`Exaone4_5_VideoProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        output = BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output


class Exaone4_5_ForConditionalGeneration(Exaone4_5_PreTrainedModel, Qwen2_5_VLForConditionalGeneration):
    config: Exaone4_5_Config

    def __init__(self, config: Exaone4_5_Config):
        super().__init__(config)
        self.model = Exaone4_5_Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        if config._num_mtp_layers > 0:
            self.mtp = Exaone4_5_MTPLayer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Exaone4_5_CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Exaone4_5_ImageProcessor.__call__`] for details. [`Exaone4_5_Processor`] uses
            [`Exaone4_5_VideoProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Exaone4_5_ForConditionalGeneration

        >>> model = Exaone4_5_ForConditionalGeneration.from_pretrained("LGAI-EXAONE/EXAONE-4.5-Beta")
        >>> processor = AutoProcessor.from_pretrained("LGAI-EXAONE/EXAONE-4.5-Beta")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        mtp_losses = []
        mtp_logits_list = []
        mtp_hidden_states_list = []
        if self.config.num_nextn_predict_layers > 0:
            mtp_labels = None
            if labels is not None:
                mtp_labels = labels.clone()

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            mtp_hidden_states = hidden_states
            for layer_idx in range(self.config.num_nextn_predict_layers):
                input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
                inputs_embeds = self.model.embed_tokens(input_ids)
                position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)
                attention_mask = create_sliding_window_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )

                mtp_hidden_states = self.mtp(
                    layer_idx=0 if self.config.mtp_share_layers else layer_idx,
                    hidden_states=mtp_hidden_states,
                    inputs_embeds=inputs_embeds,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=outputs.past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
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

        return Exaone4_5_CausalLMOutputWithPast(
            loss=loss,
            mtp_loss=mtp_losses or None,
            logits=logits,
            mtp_logits=mtp_logits_list or None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            mtp_hidden_states=mtp_hidden_states_list or None,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        # EXAONE-4.5 uses position_ids of text model, not using rope_deltas
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs


class Exaone4_5_ImageProcessor(Qwen2VLImageProcessor):
    pass


class Exaone4_5_ImageProcessorFast(Qwen2VLImageProcessorFast):
    pass


class Exaone4_5_VideoProcessor(Qwen2VLVideoProcessor):
    pass


class Exaone4_5_Processor(Qwen2_5_VLProcessor):
    tokenizer_class = ("GPT2Tokenizer", "GPT2TokenizerFast", "PreTrainedTokenizerFast")

    # def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
    #     super().__init__(image_processor, tokenizer, video_processor, chat_template, **kwargs)
    #     self.image_token = "<image_pad>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
    #     self.video_token = "<video_pad>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token


__all__ = [
    "Exaone4_5_Config",
    "Exaone4_5_TextConfig",
    "Exaone4_5_ForConditionalGeneration",
    "Exaone4_5_Model",
    "Exaone4_5_PreTrainedModel",
    "Exaone4_5_Processor",
    "Exaone4_5_ImageProcessor",
    "Exaone4_5_ImageProcessorFast",
    "Exaone4_5_VideoProcessor",
    "Exaone4_5_TextModel",
    "Exaone4_5_VisionPreTrainedModel",
]
