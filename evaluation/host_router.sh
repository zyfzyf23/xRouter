export CUDA_VISIBLE_DEVICES=6,7

# vllm serve checkpoints/Router-Tool-RL/cont-train_0902-cost-7b-lambda3-bs-16/best_global_step_5/actor/huggingface \
#    --port 8000 \
#    --max-model-len 32768 \
#    --gpu-memory-utilization 0.8 \
#    --tensor-parallel-size 2 \
#    --enable-auto-tool-choice \
#    --tool-call-parser hermes \
#    --served-model-name 

# 0821-xRouter-7b-best
vllm serve checkpoints/Router-Tool-RL/train_0902-cost-7b-lambda2-bs-16/global_step_94/actor/huggingface \
   --port 8001 \
   --max-model-len 32768 \
   --gpu-memory-utilization 0.8 \
   --tensor-parallel-size 2 \
   --enable-auto-tool-choice \
   --tool-call-parser hermes \
   --served-model-name 0902-xRouter-7b-lambda2-final
