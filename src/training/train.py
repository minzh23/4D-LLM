import os
import json
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration, HfArgumentParser
from src.qwen2_5_vl_custom import Qwen2_5_VLForConditionalGeneration
from src.qwen2_5_vl_custom import Qwen2_5_VLProcessor
from training.trainer import QwenTrainer
from training.data import make_supervised_data_module
from training.params import DataArguments, ModelArguments, TrainingArguments
from training.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
import pathlib
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl, apply_liger_kernel_to_qwen2_5_vl
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward, replace_qwen_2_with_mixed_modality_forward
from src.qwen2_5_vl_custom.depth_anything_v2.dinov2 import DINOv2
from src.qwen2_5_vl_custom.modeling_qwen2_5_vl import InterpolateMLPProjector, SpatialTemporalCoordMLP
import torch.nn as nn
import wandb

local_rank = None

def rank0_print(*args):
    if local_rank == 0 or local_rank == '0' or local_rank is None:
        print(*args)

def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names

def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad

def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)
    
    # Handle merger specifically
    merger_params = model.visual.merger.parameters()
    set_requires_grad(merger_params, training_args.tune_merger)

def configure_depth_encoder(model, training_args, compute_dtype, device):
    depth_encoder = model.depth_encoder
    depth_encoder.to(dtype=compute_dtype, device=device)

    depth_encoder_params = model.depth_encoder.parameters()
    set_requires_grad(depth_encoder_params, not training_args.freeze_depth_encoder)

def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except Exception:
        return False
    
def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        wandb.init(
            project=training_args.wandb_project,
            name=training_args.wandb_run_name,
            config={k: v for k, v in vars(training_args).items() if is_json_serializable(v)}
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"

    use_liger = training_args.use_liger
    if "Qwen2.5" in model_args.model_id:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen2_5_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
    else:
        # It monkey patches the forward to handle mixed modality inputs.
        replace_qwen_2_with_mixed_modality_forward(use_liger=use_liger)
        # This is becuase mixed-modality training monkey-patches the model forward method.
        if use_liger:
            apply_liger_kernel_to_qwen2_vl(fused_linear_cross_entropy=False)
    

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."
        
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    else:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4,8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"":training_args.device},
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=training_args.bits==4,
                load_in_8bit=training_args.bits==8,
                llm_int8_skip_modules=["visual"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            )
        ))

    if "Qwen2.5" in model_args.model_id:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            # "output/video-r1_stage1/checkpoint-6363",
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            # ignore_mismatched_sizes=True,
            **bnb_model_from_pretrained_args
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_id,
            torch_dtype=compute_dtype,
            attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa", 
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False
    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(model_to_configure, training_args, compute_dtype, training_args.device)

    depth_encoder_state_dict = torch.load('src/qwen2_5_vl_custom/depth_anything_v2/checkpoints/dinov2_only_vitb.pth', map_location='cpu', weights_only=True)
    model.depth_encoder = DINOv2(model_name='vitb')
    model.depth_encoder.load_state_dict(depth_encoder_state_dict)
    configure_depth_encoder(model_to_configure, training_args, compute_dtype, training_args.device)

    model.projector = InterpolateMLPProjector()

    model.position_embedding = SpatialTemporalCoordMLP(
        embed_dim=2048,
        hidden_dim=512,
        patch_size=2,
    )

    model.to(training_args.device)

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

    processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_id,
                                            # The default setting is padding_side="left"
                                            # When training using the right-side padding is more efficient.
                                              padding_side="right")

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(model_id=model_args.model_id,
                                              processor=processor,
                                              data_args=data_args)

    trainer = QwenTrainer(
        model=model,
        processor=processor,
        args=training_args,
        **data_module
    )

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(),  require_grad_only=False
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)
    
    wandb.finish()

if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("127.0.0.1", 5678))
    # print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached, starting execution...")
    
    # import torch.multiprocessing as mp
    # mp.set_sharing_strategy('file_system')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['NCCL_TIMEOUT'] = '1200000'  # 设置为20分钟
    os.environ['NCCL_BLOCKING_WAIT'] = '1'  # 启用阻塞等待
    train()