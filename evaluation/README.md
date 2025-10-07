# RL Router System

An intelligent LLM router system that analyzes incoming requests and routes them to the most appropriate model from a curated set of available models.

## Features

- **OpenAI-compatible API endpoints** (`/v1/chat/completions`)
- **Intelligent routing** using a specialized router model
- **Cost-optimal model selection** from multiple available models
- **Agentic prompt engineering** for each target model
- **Multi-turn conversation support**
- **Comprehensive testing suite**

## Quick Start

The RL Router system requires two steps: first hosting the router model, then serving the complete router system.

### Step 1: Host the Router Model (Port 8000)

Start the router model server that will handle routing decisions:

```bash
bash host.sh
```

This will:
- Host the router model on **port 8000**
- Configure the model with appropriate GPU settings
- Enable tool calling capabilities
- Use the Hermes chat template for tool interactions

**Wait for this to complete before proceeding to Step 2.**

### Step 2: Serve the Router System (Port 8800)

In a new terminal, start the router API server:

```bash
bash serve.sh
```

This will:
- Start the router API server on **port 8800**
- Connect to the hosted model on port 8000 for routing decisions
- Provide OpenAI-compatible endpoints
- Enable intelligent model selection and routing

## Usage

Once both services are running, you can use the router system like any OpenAI-compatible API:

### Python Client Example

```python
import openai

# Initialize client pointing to the router system
client = openai.OpenAI(
    base_url="http://localhost:8800/v1",
    api_key="dummy"  # Not required for local usage
)

# Make a request
response = client.chat.completions.create(
    model="router-tool-rl",  # Use "router-tool-rl" as thedefault model name
    messages=[
        {"role": "user", "content": "What is the best approach to solve this complex math problem: ..."}
    ],
    max_tokens=1000,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### cURL Example

```bash
curl -X POST "http://localhost:8800/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy" \
  -d '{
    "model": "router",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "max_tokens": 500
  }'
```

## Testing

The system includes comprehensive test files to verify functionality:

### Test the Hosted Model (Port 8000)

```bash
python test_host.py
```

This tests the basic functionality of the hosted router model directly.

### Test the Complete Router System (Port 8800)

```bash
python test_serve.py
```

This provides a comprehensive test suite including:
- **Simple queries** - Basic question answering
- **Math problems** - Complex mathematical reasoning
- **Coding tasks** - Programming and algorithm questions
- **Reasoning tasks** - Logic puzzles and complex reasoning
- **Multi-turn conversations** - Context-aware dialogue
- **Creative writing** - Story generation and creative tasks
- **System prompts** - Custom system prompt handling

#### Running Specific Tests

```bash
# Run all tests
python test_serve.py --test all

# Run specific test types
python test_serve.py --test math_problem
python test_serve.py --test coding_task
python test_serve.py --test reasoning_task

# Get server info
python test_serve.py --info

# List available models
python test_serve.py --models
```

## API Endpoints

The router system provides OpenAI-compatible endpoints:

- **POST** `/v1/chat/completions` - Main chat completion endpoint
- **GET** `/v1/models` - List available models
- **GET** `/health` - Health check
- **GET** `/` - Service status and configuration
- **GET** `/debug/tools` - Debug information about available tools

## Response Metadata

Responses include additional metadata about the routing process:

```json
{
  "choices": [...],
  "usage": {...},
  "router_metadata": {
    "model_used": "gpt-4o",
    "total_cost": 0.000123,
    "total_time": 1.45,
    "routing_strategy": "select_response",
    "call_history": [...],
    "turns_used": 2
  }
}
```

## How It Works

1. **Request Analysis**: The system analyzes incoming requests to understand the task complexity and type
2. **Model Selection**: Uses the router model to intelligently select the most appropriate model(s) for the task
3. **Agentic Processing**: May call multiple models and compare responses before selecting the best one
4. **Response Optimization**: Applies model-specific prompt engineering for optimal results
5. **Cost Tracking**: Monitors and reports costs for transparency

## Configuration

The system can be configured through command-line arguments in the shell scripts:

### Hosting Configuration (host.sh)
- **GPU settings**: CUDA device allocation and memory utilization
- **Model parameters**: Max model length, tensor parallel size
- **Tool settings**: Tool calling parser and chat template

### Serving Configuration (serve.sh)
- **Router model**: Which model to use for routing decisions
- **Max turns**: Maximum agentic reasoning turns (default: 3)
- **Hosted port**: Port where the router model is hosted (8000)
- **Serve port**: Port for the API server (8800)

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure the hosted model (port 8000) is fully loaded before starting the router system (port 8800)

2. **GPU Memory Issues**: Adjust `--gpu-memory-utilization` in host.sh if you encounter CUDA memory errors

3. **Port Conflicts**: Make sure ports 8000 and 8800 are available

4. **Model Loading**: The router model may take several minutes to load initially

### Logs and Debugging

- Check the terminal output from both `host.sh` and `serve.sh` for detailed logs
- Use the `/health` endpoint to verify service status
- Run `test_host.py` first to isolate hosting issues
- Use the debug endpoints (`/debug/tools`) for troubleshooting

## Architecture

```
User Request → Router API (8800) → Router Model (8000) → Target Models → Response
```

The system uses a two-tier architecture where the router model analyzes requests and selects appropriate target models for execution, providing intelligent routing with cost optimization and quality assurance. 