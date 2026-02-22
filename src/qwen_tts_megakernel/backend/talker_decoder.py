import math
from dataclasses import dataclass
from typing import Tuple

import torch

# These imports assume you have:
#   - qwen_megakernel in your Python path (from the AlpinDale repo)
#   - qwen_tts installed from PyPI / HF
from qwen_megakernel import model as mk_model
from qwen_tts.core.models import Qwen3TTSForConditionalGeneration


# Reuse config constants from the megakernel host model
NUM_LAYERS = mk_model.NUM_LAYERS
NUM_KV_HEADS = mk_model.NUM_KV_HEADS
HEAD_DIM = mk_model.HEAD_DIM
HIDDEN_SIZE = mk_model.HIDDEN_SIZE
INTERMEDIATE_SIZE = mk_model.INTERMEDIATE_SIZE
MAX_SEQ_LEN = mk_model.MAX_SEQ_LEN

# CUDA op exposed by the megakernel
_decode = torch.ops.qwen_megakernel_C.decode


def _build_rope_tables(
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build RoPE cos/sin tables as expected by the megakernel.
    """
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).to(device).contiguous()
    sin = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).to(device).contiguous()
    return cos, sin


def _pack_layer_weights(layer_weights):
    """
    Pack a list of per-layer weight tensors into a contiguous buffer of pointers.

    The megakernel host code expects a flat array of uint8 holding 64-bit addresses,
    laid out as [layer0_w0_ptr, ..., layer0_wN_ptr, layer1_w0_ptr, ...].
    """
    ptr_size = 8
    n_ptrs = 11  # number of weight pointers per layer, must match kernel expectation
    if len(layer_weights) != NUM_LAYERS * n_ptrs:
        raise ValueError(
            f"Expected {NUM_LAYERS * n_ptrs} layer weights, got {len(layer_weights)}"
        )

    buf = bytearray(NUM_LAYERS * n_ptrs * ptr_size)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            t = layer_weights[i * n_ptrs + j]
            if not t.is_cuda:
                raise ValueError("All weights must be on CUDA device before packing")
            ptr = int(t.data_ptr())
            offset = (i * n_ptrs + j) * ptr_size
            buf[offset : offset + ptr_size] = ptr.to_bytes(ptr_size, "little")

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


@dataclass
class TalkerWeights:
    embed_weight: torch.Tensor
    layer_weights_packed: torch.Tensor
    final_norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    cos_table: torch.Tensor
    sin_table: torch.Tensor
    vocab_size: int
    tokenizer: object  # Qwen tokenizer (left generic)
    eos_token_id: int


class TalkerDecoder:
    """
    Stateful single-token decoder for the Qwen3-TTS talker using the megakernel.
    """

    def __init__(self, weights: TalkerWeights, device: str = "cuda"):
        self._device = torch.device(device)
        self._weights = weights

        self._embed_weight = weights.embed_weight
        self._final_norm_weight = weights.final_norm_weight
        self._lm_head_weight = weights.lm_head_weight
        self._cos_table = weights.cos_table
        self._sin_table = weights.sin_table
        self._layer_weights_packed = weights.layer_weights_packed
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        bf16 = dict(dtype=torch.bfloat16, device=self._device)
        f32 = dict(dtype=torch.float32, device=self._device)

        self._position = 0
        self._k_cache = torch.zeros(
            (NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM), **bf16
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(16 * HEAD_DIM, **f32)
        self._k = torch.empty(8 * HEAD_DIM, **f32)
        self._v = torch.empty(8 * HEAD_DIM, **f32)
        self._attn_out = torch.empty(16 * HEAD_DIM, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device=self._device)
        self._out_token = torch.empty(1, dtype=torch.int32, device=self._device)

    @property
    def tokenizer(self):
        return self._weights.tokenizer

    @property
    def eos_token_id(self) -> int:
        return self._weights.eos_token_id

    def reset(self):
        """
        Reset KV caches and sequence position. Call before decoding a new sequence.
        """
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    def step(self, token_id: int) -> int:
        """
        Decode a single next-token step. Returns the sampled token ID (argmax).
        """
        _decode(
            self._out_token,
            int(token_id),
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            int(self._position),
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return int(self._out_token.item())

    def generate_ids(self, input_ids, max_new_tokens: int = 128) -> torch.Tensor:
        """
        Simple greedy decode loop using the megakernel for next-token prediction.
        """
        self.reset()
        # Feed existing prompt tokens through the kernel (excluding last token as next-step input)
        last_id = int(input_ids[-1])
        for tid in input_ids[:-1]:
            _ = self.step(int(tid))

        generated = [last_id]
        for _ in range(max_new_tokens):
            next_id = self.step(generated[-1])
            generated.append(next_id)
            if next_id == self.eos_token_id:
                break
        return torch.tensor(generated, device=self._device, dtype=torch.long)


def _extract_talker_backbone(tts: Qwen3TTSForConditionalGeneration):
    """
    Extract the talker backbone and tokenizer from the Qwen3-TTS model.

    This assumes the model has attributes:
      - tts.tokenizer
      - tts.talker (with .model and .config)
    and uses a Qwen3-like naming convention for layers.
    """
    tokenizer = getattr(tts, "tokenizer", None)
    talker = tts.talker  # Qwen3TTSTalkerForConditionalGeneration
    backbone = talker.model
    cfg = backbone.config

    return backbone, tokenizer, cfg


def load_talker_weights(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device: str = "cuda",
) -> TalkerWeights:
    """
    Load Qwen3-TTS talker weights into the megakernel's expected memory layout.
    """
    dev = torch.device(device)

    tts = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    backbone, tokenizer, cfg = _extract_talker_backbone(tts)

    assert cfg.hidden_size == HIDDEN_SIZE, f"hidden_size mismatch: {cfg.hidden_size}"
    assert cfg.num_hidden_layers == NUM_LAYERS, f"num_layers mismatch: {cfg.num_hidden_layers}"
    assert cfg.head_dim == HEAD_DIM, f"head_dim mismatch: {cfg.head_dim}"
    assert cfg.num_key_value_heads == NUM_KV_HEADS, (
        f"num_kv_heads mismatch: {cfg.num_key_value_heads}"
    )

    state = backbone.state_dict()

    embed_weight = state["model.embed_tokens.weight"].to(dev).contiguous()
    final_norm_weight = state["model.norm.weight"].to(dev).contiguous()
    lm_head_weight = embed_weight  # for talker we can tie to embeddings

    cos_table, sin_table = _build_rope_tables(MAX_SEQ_LEN, HEAD_DIM, dev)

    # Collect per-layer weights in the order expected by the megakernel
    layer_weights = []
    for i in range(NUM_LAYERS):
        prefix = f"model.layers.{i}."
        layer_weights.extend(
            [
                state[prefix + "input_layernorm.weight"].to(dev).contiguous(),
                state[prefix + "self_attn.q_proj.weight"].to(dev).contiguous(),
                state[prefix + "self_attn.k_proj.weight"].to(dev).contiguous(),
                state[prefix + "self_attn.v_proj.weight"].to(dev).contiguous(),
                state[prefix + "self_attn.q_norm.weight"].to(dev).contiguous(),
                state[prefix + "self_attn.k_norm.weight"].to(dev).contiguous(),
                state[prefix + "self_attn.o_proj.weight"].to(dev).contiguous(),
                state[prefix + "post_attention_layernorm.weight"].to(dev).contiguous(),
                state[prefix + "mlp.gate_proj.weight"].to(dev).contiguous(),
                state[prefix + "mlp.up_proj.weight"].to(dev).contiguous(),
                state[prefix + "mlp.down_proj.weight"].to(dev).contiguous(),
            ]
        )

    layer_weights_packed = _pack_layer_weights(layer_weights)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(cfg, "eos_token_id", 2)

    return TalkerWeights(
        embed_weight=embed_weight,
        layer_weights_packed=layer_weights_packed,
        final_norm_weight=final_norm_weight,
        lm_head_weight=lm_head_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        vocab_size=cfg.vocab_size,
        tokenizer=tokenizer,
        eos_token_id=eos_token_id,
    )


def load_talker_decoder(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device: str = "cuda",
) -> Tuple[TalkerDecoder, object]:
    """
    Convenience function: load weights and construct a TalkerDecoder + tokenizer.
    """
    weights = load_talker_weights(model_name=model_name, device=device)
    decoder = TalkerDecoder(weights, device=device)
    return decoder, weights.tokenizer