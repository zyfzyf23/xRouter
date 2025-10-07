# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score


async def single_compute_score(compute_score_fn, data_source, solution_str, ground_truth, extra_info, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(compute_score_fn, data_source=data_source, solution_str=solution_str, 
                            ground_truth=ground_truth, extra_info=extra_info)
                ),
                timeout=timeout,
            )
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for solution: {solution_str[:64]}...")
        return None
    except Exception as e:
        print(f"Error processing solution: {solution_str[:10]}, Error: {e}")
        return None


async def parallel_compute_score_async(compute_score_fn, data_sources, solutions, ground_truths, 
                                      extra_infos, num_processes=64, batch_size=None, shuffle=False):
    # If batch_size is not set, process all items at once
    if batch_size is None or batch_size <= 0:
        batch_size = len(data_sources)
    
    # Create indices for tracking original positions
    indices = list(range(len(data_sources)))
    
    # Shuffle data if required
    if shuffle:
        # Create a copy of the original indices for restoring order later
        original_indices = indices.copy()
        # Create shuffled indices
        shuffled_indices = np.random.permutation(len(data_sources))
        
        # Apply shuffling to all data arrays
        data_sources = [data_sources[i] for i in shuffled_indices]
        solutions = [solutions[i] for i in shuffled_indices]
        ground_truths = [ground_truths[i] for i in shuffled_indices]
        extra_infos = [extra_infos[i] for i in shuffled_indices]
        # Map shuffled positions to original indices
        indices = [original_indices[i] for i in shuffled_indices]
    
    results = [None] * len(data_sources)
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Process data in batches
        for start_idx in range(0, len(data_sources), batch_size):
            end_idx = min(start_idx + batch_size, len(data_sources))
            
            # Create tasks for current batch
            tasks_async = [
                single_compute_score(
                    compute_score_fn, 
                    data_sources[i], 
                    solutions[i], 
                    ground_truths[i], 
                    extra_infos[i], 
                    executor, 
                    timeout=300.
                )
                for i in range(start_idx, end_idx)
            ]
            
            # Handle potential exceptions to prevent process starvation
            try:
                batch_results = await asyncio.gather(*tasks_async, return_exceptions=False)
                
                # Store results in their correct positions
                for i, result in enumerate(batch_results):
                    actual_idx = start_idx + i
                    results[actual_idx] = result
                    
            except Exception as e:
                for pid, proc in executor._processes.items():
                    try:
                        proc.kill()
                    except Exception as kill_err:
                        print('shut down failed: ' + str(kill_err))
                raise
    
    # Restore original order if data was shuffled
    if shuffle:
        # Create a mapping to restore original order
        ordered_results = [None] * len(results)
        for i, original_idx in enumerate(indices):
            ordered_results[original_idx] = results[i]
        results = ordered_results
    
    return results


class AsyncDAPORewardManager:
    """The reward manager.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 compute_score=None,
                 reward_fn_key='data_source',
                 max_resp_len=None,
                 overlong_buffer_cfg=None,
                 batch_size=2048,
                 shuffle_batch=True,
                 **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.batch_size = batch_size
        self.shuffle_batch = shuffle_batch

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False, reward_lambda: float = 3.0, reward_K: float = 1.0, cost_max: float = 0.8, simple_mode: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        # print(f"[DEBUG] data.batch['responses'] shape: {data.batch['responses'].shape}")
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # print(f"[DEBUG] reward_tensor initial shape: {reward_tensor.shape}")

        # Add this to understand DataProto structure
        # print(f"[DEBUG] DataProto length: {len(data)}")
        # print(f"[DEBUG] DataProto batch keys: {list(data.batch.keys())}")
        # print(f"[DEBUG] DataProto non_tensor_batch keys: {list(data.non_tensor_batch.keys())}")

        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # print(f"[DEBUG] Processing {len(data)} items")

        # Count data sources
        data_source_counts = defaultdict(int)
        for i in range(len(data)):
            data_source = data[i].non_tensor_batch[self.reward_fn_key]
            data_source_counts[data_source] += 1
        
        # print(f"[DEBUG] Data source distribution: {dict(data_source_counts)}")

        # Check if any data is being filtered
        for i in range(len(data)):
            data_item = data[i]
            if self.reward_fn_key not in data_item.non_tensor_batch:
                # print(f"[DEBUG] Warning: Item {i} missing reward_fn_key '{self.reward_fn_key}'")
                pass
            
            # Check if ground truth exists
            if 'reward_model' not in data_item.non_tensor_batch or 'ground_truth' not in data_item.non_tensor_batch['reward_model']:
                # print(f"[DEBUG] Warning: Item {i} missing ground_truth")
                pass

        # Prepare data for parallel processing
        data_sources = []
        solutions = []
        ground_truths = []
        extra_infos = []
        valid_response_lengths = []
        prompt_strs = []
        response_strs = []

        # additional customized data
        costs_distribution = {}
        tool_num_distribution = {}
        turn_distribution = {}

        failure_distribution = {}
        route_distribution = {}

        all_cost = []
        all_tool_num = []
        all_turn = []
        all_valid = []
        all_normal_route_user = []
        all_tool_route_user = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_lengths.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            prompt_strs.append(prompt_str)
            
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]
            response_strs.append(response_str)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            data_sources.append(data_source)
            # solutions.append(response_str)
            ground_truths.append(ground_truth)
            extra_infos.append(extra_info)

            # TODO: Directly use the last turn in the conversation as the response
            conversation_histories = data_item.non_tensor_batch["conversation_histories"]
            if type(conversation_histories) == str:
                conversation_histories = json.loads(conversation_histories)

            if "simple" not in data_source.lower() and not simple_mode:
                # Make sure that route back to user is a single tool call as the last turn
                if len(conversation_histories[-1]["route_to"]) == 1 and conversation_histories[-1]["route_to"][0] == "user":
                    solutions.append(conversation_histories[-1]["feedback"][-1])
                    all_valid.append(1.0)
                    # content is [] indicates the model's giving a normal response
                    if len(conversation_histories[-1]["content"]) == 0:
                        all_normal_route_user.append(1.0)
                        all_tool_route_user.append(0.0)
                    else:
                        all_normal_route_user.append(0.0)
                        all_tool_route_user.append(1.0)
                else:
                    solutions.append("No valid solution found!")
                    all_valid.append(0.0)
                    all_normal_route_user.append(0.0)
                    all_tool_route_user.append(0.0)
            elif "simple" not in data_source.lower() and simple_mode:
                solution_found = False
                for feedback, metadata in zip(conversation_histories[-1]["feedback"], conversation_histories[-1]["metadata"]):
                    if metadata["success"]:
                        solution_found = True
                        solutions.append(feedback)
                        all_valid.append(1.0)
                        break
                if not solution_found:
                    solutions.append("No valid solution found!")
                    all_valid.append(0.0)
                # we do not make statistics about normal route to user and tool route to user for simple mode
                all_normal_route_user.append(0.0)
                all_tool_route_user.append(0.0)
            else: # for simple data source, we check if the tool is found (no matter simple mode or not)
                tool_found = False
                for turn in conversation_histories:
                    for route_to in turn["route_to"]:
                        if route_to != "user":
                            tool_found = True
                            break
                if tool_found:
                    solutions.append("Wrong! Tool is found!")
                else:
                    solutions.append("Correct! No tool is found!")
                all_valid.append(1.0)
                if len(conversation_histories[-1]["content"]) == 0:
                    all_normal_route_user.append(1.0)
                    all_tool_route_user.append(0.0)
                else:
                    all_normal_route_user.append(0.0)
                    all_tool_route_user.append(1.0)
            
            # statistics on failure modes
            for turn in conversation_histories:
                if "route_to" in turn and "metadata" in turn and len(turn["route_to"]) == len(turn["metadata"]):
                    for route_to, metadata in zip(turn["route_to"], turn["metadata"]):
                        if route_to == "user" or "success" not in metadata:
                            continue
                        if route_to not in route_distribution:
                            route_distribution[route_to] = 0
                        route_distribution[route_to] += 1
                        if metadata["success"] == False:
                            if route_to not in failure_distribution:
                                failure_distribution[route_to] = 0
                            failure_distribution[route_to] += 1
            
            # additional customized data
            local_cost = 0
            local_tool_num = 0
            for turn in conversation_histories:
                if "metadata" in turn:
                    metadata_list = turn["metadata"]
                    for metadata in metadata_list:
                        if "cost" in metadata:
                            local_cost += metadata["cost"]
                if "route_to" in turn:
                    for route_to in turn["route_to"]:
                        if route_to != "user":
                            local_tool_num += 1
            local_turn = len(conversation_histories)
            
            if data_source not in costs_distribution:
                costs_distribution[data_source] = []
            if data_source not in tool_num_distribution:
                tool_num_distribution[data_source] = []
            if data_source not in turn_distribution:
                turn_distribution[data_source] = []
            costs_distribution[data_source].append(local_cost)
            tool_num_distribution[data_source].append(local_tool_num)
            turn_distribution[data_source].append(local_turn)

            all_cost.append(local_cost)
            all_tool_num.append(local_tool_num)
            all_turn.append(local_turn)

        reward_extra_info["customized_avg_cost"] = all_cost
        reward_extra_info["customized_avg_tool_num"] = all_tool_num
        reward_extra_info["customized_avg_turn"] = all_turn
        reward_extra_info["customized_avg_valid"] = all_valid
        reward_extra_info["customized_avg_normal_route_user"] = all_normal_route_user
        reward_extra_info["customized_avg_tool_route_user"] = all_tool_route_user

        for data_source in costs_distribution:
            reward_extra_info[f"customized_{data_source}_avg_cost"] = costs_distribution[data_source]
        for data_source in tool_num_distribution:
            reward_extra_info[f"customized_{data_source}_avg_tool_num"] = tool_num_distribution[data_source]
        for data_source in turn_distribution:
            reward_extra_info[f"customized_{data_source}_avg_turn"] = turn_distribution[data_source]

        for route_to in route_distribution:
            if route_to not in failure_distribution:
                failure_distribution[route_to] = 0
            failure_percentage = round(failure_distribution[route_to] / route_distribution[route_to], 4)
            reward_extra_info[f"failure_{route_to}_fail_percentage"] = [failure_percentage]
            reward_extra_info[f"failure_{route_to}_fail_count"] = [failure_distribution[route_to]]
            reward_extra_info[f"failure_{route_to}_route_count"] = [route_distribution[route_to]]

        # Run parallel score computation
        try:
            # print(f"[DEBUG] Starting parallel score computation for {len(solutions)} items")
            results = asyncio.run(
                parallel_compute_score_async(
                    self.compute_score,
                    data_sources,
                    solutions,
                    ground_truths,
                    extra_infos,
                    num_processes=64,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle_batch
                )
            )
            # print(f"[DEBUG] Parallel score computation completed")
        except Exception as e:
            # print(f"[DEBUG] Error in parallel score computation: {e}")
            # Fallback to zeros if computation fails
            results = [None] * len(solutions)

        # other customized data
        all_original_reward = []
        all_cost_reward = []
        original_reward_distribution = {}

        # Process results
        for i, (result, data_source, response_str, ground_truth, valid_response_length, cost) in enumerate(
            zip(results, data_sources, response_strs, ground_truths, valid_response_lengths, all_cost)
        ):
            score = 0.0
            if result is None:
                result = {"score": 0.0, "acc": 0.0}
            else:
                result = result[0]  # Unwrap from asyncio.gather
                if not isinstance(result, dict):
                    # Hack to avoid some rewards don't return a dict
                    result = {"score": result, "acc": result}
            # Minimal special handling for simple data source
            if "simple" in str(data_source).lower():
                sol = solutions[i] if i < len(solutions) else ""
                simple_score = 1.0 if "Correct!" in sol else 0.0
                result = {"score": simple_score, "acc": simple_score}
            
            score = result["score"]            
            # reward = score

            # Begin cutmization with cost consideration
            all_original_reward.append(score)
            if data_source not in original_reward_distribution:
                original_reward_distribution[data_source] = []
            original_reward_distribution[data_source].append(score)

            effective_cost_penalty = min(reward_lambda * cost, cost_max)
            reward = score * (reward_K - effective_cost_penalty)
            result = {"score": reward, "acc": reward}
            all_cost_reward.append(reward)

            # Store the information including original reward
            for key, value in result.items():
                # print(f"[DEBUG] in reward_extra_info, key: {key}, value: {value}")
                reward_extra_info[key].append(value)

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            # print(f"[DEBUG] Item {i}, valid_response_length: {valid_response_length}, reward: {reward}")
            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_strs[i])
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if result is not None and isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        # print(f"[DEBUG] Final reward_tensor shape: {reward_tensor.shape}")
        # print(f"[DEBUG] Non-zero elements in reward_tensor: {(reward_tensor != 0).sum().item()}")
        # print(f"[DEBUG] Unique data sources processed: {list(already_print_data_sources.keys())}")

        reward_extra_info["customized_original_reward"] = all_original_reward
        reward_extra_info["customized_cost_reward"] = all_cost_reward
        for data_source in original_reward_distribution:
            reward_extra_info[f"customized_{data_source}_avg_original_reward"] = original_reward_distribution[data_source]
        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor