# DeepSpeed Integration for VSI Benchmark Evaluation

This guide explains how to use DeepSpeed to reduce memory usage when evaluating the VSI Benchmark with Qwen2.5-VL models.

## Installation Requirements

First, install DeepSpeed:
```bash
pip install deepspeed
```

## Usage Options

### 1. Quick Start with Default Settings

Enable DeepSpeed with default configuration:
```bash
python evaluate_vsi.py --use_deepspeed --cpu_offload
```

### 2. Using Custom Configuration Files

Use a pre-configured DeepSpeed setup:
```bash
# Maximum memory savings (ZeRO Stage 3 + CPU offload)
python evaluate_vsi.py --use_deepspeed --deepspeed_config deepspeed_inference_config.json

# Faster inference (ZeRO Stage 2, no CPU offload)
python evaluate_vsi.py --use_deepspeed --deepspeed_config deepspeed_inference_config_light.json
```

### 3. Multi-GPU Setup

For multiple GPUs:
```bash
deepspeed --num_gpus=2 evaluate_vsi.py --use_deepspeed --deepspeed_config deepspeed_inference_config.json
```

### 4. Using the Provided Script

Run the pre-configured evaluation script:
```bash
./run_evaluation_with_deepspeed.sh
```

## Configuration Files

### `deepspeed_inference_config.json`
- **ZeRO Stage 3** with CPU offloading
- **Maximum memory savings**
- Slower inference due to CPU-GPU data movement
- Recommended for very large models or limited GPU memory

### `deepspeed_inference_config_light.json`
- **ZeRO Stage 2** without CPU offloading
- **Balanced memory usage and speed**
- Faster inference than Stage 3
- Good for models that fit in GPU memory with some optimization

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--use_deepspeed` | Enable DeepSpeed integration | False |
| `--cpu_offload` | Enable CPU parameter offloading | False |
| `--deepspeed_config` | Path to DeepSpeed config file | None (uses default) |

## Memory Optimization Features

When DeepSpeed is enabled, the following optimizations are applied:

1. **ZeRO Optimization**: Model parameters are partitioned across GPUs
2. **CPU Offloading**: Parameters can be stored in CPU memory
3. **Enhanced Memory Cleanup**: More aggressive garbage collection
4. **Gradient Management**: Explicit gradient disabling during inference
5. **Memory Monitoring**: Peak memory statistics tracking

## Expected Memory Savings

Typical memory reductions with DeepSpeed:

- **ZeRO Stage 2**: 2-4x memory reduction
- **ZeRO Stage 3**: 4-8x memory reduction  
- **ZeRO Stage 3 + CPU Offload**: 8-16x memory reduction

*Note: Actual savings depend on model size, batch size, and hardware configuration.*

## Performance Considerations

- **ZeRO Stage 2**: Minimal performance impact
- **ZeRO Stage 3**: 10-20% slower due to parameter gathering
- **CPU Offload**: 20-50% slower due to CPU-GPU transfers

Choose the configuration based on your memory constraints and performance requirements.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Try enabling `--cpu_offload` or reducing `--batch_size`
2. **Slow Performance**: Use `deepspeed_inference_config_light.json` instead
3. **Multi-GPU Issues**: Ensure all GPUs are available and properly configured

### Monitoring Memory Usage

Monitor GPU memory during evaluation:
```bash
watch -n 1 nvidia-smi
```

### Debug Mode

For detailed DeepSpeed logging:
```bash
export DEEPSPEED_LOG_LEVEL=INFO
python evaluate_vsi.py --use_deepspeed
``` 