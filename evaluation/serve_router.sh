# CUDA_VISIBLE_DEVICES=0 vllm serve checkpoints/3b-train_0819-cost-3b-cont-lambda5-fast-mini-bsz-16-step-38 \
#     --port 8000 \
#     --host 0.0.0.0 \
#     --served-model-name router-tool-rl \
#     --tensor-parallel-size 1 \
#     --tool-call-parser hermes \
#     --enable-auto-tool-choice


# run host.sh to start the router server first.

# python serve_router.py \
#    --router-model 0821-xRouter-7b-best \
#    --max-turns 3 \
#    --hosted-port 8000 \
#    --port 8800

python serve_router.py \
   --router-model 0902-xRouter-7b-lambda2-final \
   --max-turns 3 \
   --hosted-port 8001 \
   --port 8801

# simple mode
# python serve_router_simple_mode.py \
#    --router-model xRouter-7b \
#    --hosted-port 8000 \
#    --port 8800