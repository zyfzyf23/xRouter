import csv
import numpy as np
import os
import pandas as pd
from evaluation.test_serve import RouterTestClient
import argparse
import json
from tqdm import tqdm
import ast
import openai
from openai import OpenAI
import time
import litellm
from serve_router import calculate_router_cost
from verl.tools.utils.router_utils import MODEL_SPECS

# {'content': 'According to the calculations, the total time Aya spends on her walk, including the time spent in the coffee shop, is \\(\\boxed{204}\\) minutes.', 
# 'response_time': 16.569101095199585, 
# 'usage': {'completion_tokens': 35, 'prompt_tokens': 147, 'total_tokens': 182, 'completion_tokens_details': None, 'prompt_tokens_details': None}, 
# 'router_metadata': {'model_used': None, 'total_cost': 0.004772900000000001, 'total_time': 15.864969968795776, 'routing_strategy': 'direct_response', 'call_history': [{'call_id': 'chatcmpl-tool-099980eb29ea46aab28fdf2ceb71ef5c', 'model_id': 'o4-mini', 'function_name': 'call_o4_mini', 'arguments': {'optimized_system_prompt': 'Solve
#  the problem: Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of s kilometers per hour, the walk 
# takes her 4 hours, including t minutes spent in the coffee shop. When she walks s+2 kilometers per hour, the walk takes her 2 hours and 24 minutes, including t minutes spent 
# in the coffee shop. Suppose Aya walks at s+1/2 kilometers per hour. Find the number of minutes the walk takes her, including the t minutes spent in the coffee shop.', 'temper
# ature': 0.5}, 'response': 'Let the coffee‐shop stop be t minutes long.  In hours the stop is t/60.  Then\n\n(1) 9/s + t/60 = 4  \n(2) 9/(s+2) + t/60 = 2\u2009+\u200924/60 = 2
# .4.  \n\nSubtracting (2) from (1):  \n4 − 2.4 = 9/s − 9/(s+2)  \n1.6 = 9·[(s+2) − s]/[s(s+2)] = 18/[s(s+2)]  \n⇒ s(s+2) = 18/1.6 = 11.25  \n⇒ s² + 2s − 11.25 = 0  \n⇒ (using 
# the quadratic formula) s = 2.5\u2009km/h (rejecting the negative root).\n\nThen from (1):  \nt/60 = 4 − 9/2.5 = 4 − 3.6 = 0.4  ⇒ t = 0.4·60 = 24\u2009minutes.\n\nFinally, at 
# speed s+½ = 2.5+0.5 = 3\u2009km/h the walk time is 9/3 = 3\u2009h = 180\u2009min, plus the 24\u2009min stop gives\n\nAnswer: 180 + 24 = 204\u2009minutes.  ', 'cost': 0.004772
# 900000000001, 'tokens_used': 1279, 'latency': 11.910168647766113}]}}

client_single = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add the select_reward_fn from main_eval.py
def select_reward_fn(data_source):
    if data_source == 'lighteval/MATH':
        from verl.utils.reward_score import math
        return math.compute_score
    else:
        # install rllm from https://github.com/rllm-org/rllm
        from rllm.rewards.rl_reward import rllm_reward_fn
        return rllm_reward_fn

def main(args):
    
    client = RouterTestClient(args.url)
    client_single = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if args.single_model:
        output_filepath = args.output_path + "/outputs/"+os.path.basename(args.data_path).split('.')[0] + "_" + args.model_path + "_single.json"
    else:
        output_filepath = args.output_path + "/outputs/"+os.path.basename(args.data_path).split('.')[0] + "_" + args.model_path + ".json"

    import json
    # load only questions without responses
    
    predictions = {}
    
    # Handle specific requests
    if args.models:
        print("\n📋 Available Models:")
        models = client.get_available_models()
        if "models" in models:
            for model in models["models"]:
                print(f"  - {model}")
        else:
            print(f"❌ Error: {models.get('error', 'Unknown error')}")
        return
    
    if args.info:
        print("\n🔍 Server Information:")
        info = client.get_server_info()
        print(json.dumps(info, indent=2))
        return
    
    try:
        dataset = pd.read_parquet(args.data_path)
        chat_lst = dataset["prompt"].tolist()
        chat_lst = [chat.tolist() for chat in chat_lst]
        data_sources = dataset["data_source"].tolist()
        reward_model_data = dataset["reward_model"].tolist()
        
    except Exception as e:
        # Read json
        import json
        args.data_path = args.data_path.replace('.parquet', '.json')
        with open(args.data_path, 'r') as f:
            dataset = pd.read_json(f)
        chat_lst = dataset["prompt"].tolist()
        chat_lst = [ast.literal_eval(chat) for chat in chat_lst]
    passes = 0
    total = len(dataset)
    total_scores = []
    total_response_time = 0
    total_tokens = 0
    total_cost = 0
    total_count = 0
    for i, prompt in tqdm(enumerate(chat_lst), total=len(chat_lst)):
        data_source = data_sources[i]
        reward_data = reward_model_data[i]
        reward_fn = select_reward_fn(data_source)
        ground_truth = reward_data['ground_truth']
        
        score_lst = []
        predictions[i] = {}
        for j in range(args.num_rollouts):
            prompt = [{"role": "user", "content": prompt[0]['content'] + " " +prompt[1]['content']}]
            
            for k in range(5):  
                try:
                    start_time = time.time()
                    if not args.single_model:
                        response = client.chat_completion(prompt, model=args.model_path)
                    else:
                        spec = MODEL_SPECS[args.model_path]
                        response = litellm.completion(model=spec.api_alias, messages=prompt, temperature=1.0, max_completion_tokens=32768)
                    if 'usage' in response or response.choices[0].message.content:
                        end_time = time.time()
                        response_time = end_time - start_time
                        break
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            if not args.single_model:
                total_response_time += response['response_time']
                total_tokens += response['usage']['total_tokens']
                total_cost += response['router_metadata']['total_cost']
                response_content = response['content']
            else:
                total_response_time += response_time
                total_tokens += response.usage.completion_tokens
                total_cost += calculate_router_cost(args.model_path, response.usage.prompt_tokens, response.usage.completion_tokens)
                response_content = response.choices[0].message.content
                
            total_count += 1
            
            try:
                score = reward_fn(data_source, response_content, ground_truth, {"use_format_reward": args.use_format_reward})
                score_lst.append(score)
            except Exception as e:
                score = reward_fn(response_content, ground_truth, {"use_format_reward": args.use_format_reward})
                score_lst.append(score)
                
            if not args.single_model:
                predictions[i][j] = {
                "model": args.model_path,
                "available_models": client.get_available_models(),
                "response": response,
                "response_time": response['response_time'],
                    "tokens": response['usage']['total_tokens'],
                    "cost": response['router_metadata']['total_cost'],
                    "score": score
            }
            else:
                predictions[i][j] = {
                    "model": args.model_path,
                    "available_models": args.model_path,
                    "response": response,
                    "response_time": response_time,
                    "tokens": response.usage.completion_tokens,
                    "cost": total_cost,
                    "score": score
                }
                
        
        max_score = np.max(score_lst)
        total_scores.append(score_lst)
        if max_score == 1:
            passes += 1
    pass_at_n = passes / total
    pass_at_1 = np.mean(total_scores)
    
    # Save metrics to CSV
    csv_path = os.path.join(args.output_path, f'pass_{data_source.split("-")[0]}_num_rollouts_{args.num_rollouts}.csv')
    
    # Prepare the row data
    # Extract the dataset name from the path
    # if args.model_path is None:
    #     args.model_path = "router"
    if args.single_model:
        model_path = args.model_path + "_single"
    else:
        model_path = args.model_path
        
    dataset_name = os.path.basename(args.data_path).split('.')[0]
    row_data = {
        'model_path': model_path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        # f'pass@{args.num_rollouts}': pass_at_n,
        'avg_response_time': total_response_time/total_count,
        'avg_tokens': total_tokens/total_count,
        'avg_cost': total_cost/total_count,
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
        
    # cache responses
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)
        
    return pass_at_1, pass_at_n

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Router API Test Client")
    parser.add_argument("--url", default="http://localhost:8806/v1", 
                       help="Base URL of the router API")
    parser.add_argument("--data_path", default="/fsx/home/skokane/research/verl_mt/data/deepscaler/no_think/aime.parquet", 
                       help="Path to the data file")
    parser.add_argument("--output_path", default="/fsx/home/skokane/research/rl_router/checkpoints/", 
                       help="Path to the output file")
    parser.add_argument("--model_path", default="0914-xRouter-7b-lambda1-best", 
                       help="Path to the model file")
    parser.add_argument("--use_format_reward", default=False, type=bool, 
                       help="Use format reward")
    parser.add_argument("--single_model", default=False, type=bool, 
                       help="single model")
    parser.add_argument("--num_rollouts", default=1, 
                       help="Number of rollouts")
    parser.add_argument("--models", action="store_true", 
                       help="List available models")
    parser.add_argument("--info", action="store_true",
                       help="Get server information")
    
    args = parser.parse_args()
    
    import multiprocessing as mp
    mp.set_start_method('spawn')
    
    import copy
    
    arg1 = copy.copy(args)
    arg2 = copy.copy(args)
    arg3 = copy.copy(args)
    arg4 = copy.copy(args)
    arg5 = copy.copy(args)
    arg6 = copy.copy(args)
    
    
    arg1.data_path = "/fsx/home/skokane/research/verl_mt/data/deepcoder/test_humanevalplus.parquet"
    arg2.data_path = "/fsx/home/skokane/research/verl_mt/data/deepscaler/no_think/gsm8k.parquet"
    
    try:
        with mp.Pool(processes=2) as pool:
            pool.map(main, [arg1, arg2])
    except:
        pass  
    # #clear pool
    pool.close()
    pool.join()
    