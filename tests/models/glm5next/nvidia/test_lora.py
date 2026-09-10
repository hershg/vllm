"""LoRA metadata for the native GLM-5.3-Flash vLLM model."""

from vllm.model_executor.models.interfaces import supports_lora
from vllm.models.glm5next.nvidia.model import Glm5NextForCausalLM


def test_glm5_next_declares_all_packed_lora_modules():
    """HF-format updates must address every fused projection used by GLM Flash."""
    assert supports_lora(Glm5NextForCausalLM)
    assert Glm5NextForCausalLM.packed_modules_mapping == {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "in_proj_qkvbfg_a": [
            "q_proj",
            "k_proj",
            "v_proj",
            "b_proj",
            "f_a_proj",
            "g_a_proj",
        ],
    }
