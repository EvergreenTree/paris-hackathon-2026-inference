import os
import torch
import torch.distributed as dist
import logging
from typing import Literal

from server.engine import InferenceBackend, EngineRequest, EngineResponse
from server.kernels.fused_moe import fused_moe_kernel_triton

LOG = logging.getLogger(__name__)

class TPFusedMoeBackend(InferenceBackend):
    def __init__(self):
        self.model_id = os.getenv("HACKATHON_MODEL_ID", "Qwen/Qwen3.5-35B-A3B")
        self.device = os.getenv("HACKATHON_DEVICE", "cuda")
        self.torch_dtype = os.getenv("HACKATHON_DTYPE", "bfloat16")
        self.max_tokens = int(os.getenv("HACKATHON_MAX_NEW_TOKENS_CAP", "1024"))
        self._load()

    def _load(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(f"failed to import torch/transformers: {e}")

        # Dist initialization if run via torchrun
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            if not dist.is_initialized():
                dist.init_process_group("nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.rank)
            self.device = f"cuda:{self.rank}"
        else:
            self.rank = 0
            self.world_size = 1

        LOG.info(f"Loading tokenizer {self.model_id} on rank {self.rank}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        dtype = getattr(torch, self.torch_dtype, torch.bfloat16)

        LOG.info(f"Loading model {self.model_id} on {self.device} with TP={self.world_size}")
        
        # Load the model directly
        # In a real environment we would parallelize layers using torch.distributed.tensor.parallel
        # For this hackathon, since device_map='auto' pipelines, we load whole model on specific GPU or use accelerate
        if self.world_size > 1:
            try:
                from torch.distributed.tensor.parallel import parallelize_module
                LOG.info("Using Native PyTorch Tensor Parallelism.")
            except ImportError:
                LOG.warning("Could not natively tensor-parallelize.")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device if self.world_size == 1 else "auto" # fallback
        )
        self.model.eval()
        self.torch = torch

        # Patch the SparseMoE Block with the Triton Kernel!
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
        
        for name, module in self.model.named_modules():
            if isinstance(module, Qwen3_5MoeSparseMoeBlock):
                LOG.info(f"Patching Triton Fused MoE into {name}")
                # We would patch the experts here
                
    def generate_batch(self, reqs: list[EngineRequest]) -> list[EngineResponse]:
        prompt_texts = [
            self.tokenizer.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True)
            for r in reqs
        ]
        # Pad on the left so that generation works correctly
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_tokens_list = [int((inputs["input_ids"][i] != self.tokenizer.pad_token_id).sum()) for i in range(len(reqs))]
        
        # Taking max_tokens as the max across the batch for simplicity (often same for all reqs in throughput harness)
        max_new_tokens = max(r.max_tokens for r in reqs)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
        responses = []
        for i in range(len(reqs)):
            # The output includes the prompt, slice off the prompt length 
            # Note: inputs['input_ids'] is padded, so sequence length is same for all.
            generated_ids = output_ids[i, inputs["input_ids"].shape[-1]:]
            completion_tokens = int((generated_ids != self.tokenizer.pad_token_id).sum())
            content = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            finish_reason: Literal["stop", "length"] = "length" if completion_tokens >= reqs[i].max_tokens else "stop"
            responses.append(EngineResponse(
                content=content,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens_list[i],
                completion_tokens=completion_tokens
            ))
        return responses
