export CUDA_VISIBLE_DEVICES=4,5,6,7

# vllm serve model/Qwen2.5-7B-Instruct \
#    --port 8901 \
#    --max-model-len 32768 \
#    --gpu-memory-utilization 0.85 \
#    --tensor-parallel-size 4 \
#    --enable-auto-tool-choice \
#    --tool-call-parser hermes \
#    --served-model-name Qwen2.5-7B

vllm serve model/Qwen3-32B \
   --port 8900 \
   --max-model-len 32768 \
   --gpu-memory-utilization 0.85 \
   --tensor-parallel-size 4 \
   --enable-auto-tool-choice \
   --tool-call-parser hermes \
   --reasoning-parser qwen3 \
   --served-model-name Qwen3-32B
