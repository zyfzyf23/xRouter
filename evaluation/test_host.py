import litellm

# 你的本地 vLLM 地址
BASE_URL = "http://localhost:8000/v1"

# 定义参数
router_params = {
    "model": "openai/router-tool-rl",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Please introduce yourself."}
    ],
    "max_tokens": 512,
    "temperature": 0.0,
    "base_url": BASE_URL,
    "api_key": "dummy"
}

# 调用 litellm
response = litellm.completion(**router_params)

print(response["choices"][0]["message"]["content"])
