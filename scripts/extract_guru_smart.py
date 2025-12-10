#!/usr/bin/env python3
"""
Extract Math and Code prompts from LLM360/guru-RL-92k dataset.
Smart approach that loads the dataset info first to understand parquet file structure.
"""

import os
import json
from typing import List, Dict, Optional, Set
from tqdm import tqdm
import datasets

# Set HF endpoint for China network
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_PROGRESS_BARS"] = "1"


def get_dataset_info():
    """
    Get dataset info to understand the file structure.
    """
    try:
        # Get dataset info without loading the data
        info = datasets.load_dataset_builder("LLM360/guru-RL-92k")
        print("\nDataset Info:")
        print(f"Description: {info.info.description[:500] if info.info.description else 'No description'}...")

        # Try to get split info
        print("\nAvailable splits:")
        for split_name, split_info in info.splits.items():
            print(f"  - {split_name}: {split_info.num_examples if hasattr(split_info, 'num_examples') else 'Unknown size'} examples")

        return info
    except Exception as e:
        print(f"Could not get dataset info: {e}")
        return None


def explore_dataset_structure():
    """
    Try to understand the dataset structure by loading a small sample.
    """
    try:
        print("\nExploring dataset structure...")

        # Try to load just a few samples
        ds = datasets.load_dataset(
            "LLM360/guru-RL-92k",
            split="train",
            streaming=True
        )

        # Collect info from first 1000 samples
        abilities = set()
        data_sources = set()
        domain_counts = {"math": 0, "code": 0, "other": 0}
        samples_seen = 0
        target_samples = 1000

        print(f"\nAnalyzing first {target_samples} samples to understand structure...")

        for example in tqdm(ds, total=target_samples, desc="Analyzing"):
            if samples_seen >= target_samples:
                break

            samples_seen += 1

            ability = example.get("ability", "")
            data_source = example.get("data_source", "")

            abilities.add(ability)
            data_sources.add(data_source)

            # Classify domain
            if ability == "codegen" or "leetcode" in str(data_source).lower():
                domain_counts["code"] += 1
            elif "math" in str(ability).lower() or "gsm8k" in str(data_source).lower():
                domain_counts["math"] += 1
            else:
                domain_counts["other"] += 1

        print(f"\nDomain distribution in first {target_samples} samples:")
        for domain, count in domain_counts.items():
            print(f"  {domain}: {count} ({count/target_samples*100:.1f}%)")

        print(f"\nUnique abilities found: {sorted(abilities)}")
        print(f"Unique data sources found: {sorted(data_sources)[:10]}...")  # Show first 10

        return domain_counts, abilities, data_sources

    except Exception as e:
        print(f"Error exploring dataset: {e}")
        return {}, set(), set()


def extract_samples_smart():
    """
    Extract samples with smart logic based on exploration results.
    """
    print("\n" + "=" * 60)
    print("SMART SAMPLE EXTRACTION")
    print("=" * 60)

    # First, explore the structure
    domain_counts, abilities, data_sources = explore_dataset_structure()

    if not domain_counts:
        print("Could not explore dataset structure. Using fallback approach.")
        return

    # Determine which domains are available
    available_domains = [d for d, c in domain_counts.items() if c > 0]
    print(f"\nAvailable domains: {available_domains}")

    # Load the dataset for extraction
    try:
        ds = datasets.load_dataset(
            "LLM360/guru-RL-92k",
            split="train",
            streaming=True
        )

        # Set targets based on availability
        TARGET_PER_DOMAIN = 500
        collected_samples = {
            "math": [],
            "code": [],
            "other": []
        }

        print(f"\nExtracting up to {TARGET_PER_DOMAIN} samples per available domain...")

        # Process samples
        for example in tqdm(ds, desc="Extracting samples"):
            # Check if we have enough of all available domains
            all_complete = all(
                len(collected_samples[d]) >= TARGET_PER_DOMAIN or domain_counts.get(d, 0) == 0
                for d in available_domains
            )

            if all_complete:
                break

            ability = example.get("ability", "")
            data_source = example.get("data_source", "")
            prompt = example.get("prompt", [])
            response = example.get("response", "")
            completion = example.get("completion", "")

            # Classify domain
            if ability == "codegen" or "leetcode" in str(data_source).lower():
                domain = "code"
            elif "math" in str(ability).lower() or "gsm8k" in str(data_source).lower():
                domain = "math"
            else:
                domain = "other"

            # Skip if we don't need this domain or already have enough
            if domain not in available_domains or len(collected_samples[domain]) >= TARGET_PER_DOMAIN:
                continue

            # Extract question and answer
            question = extract_user_content_from_chat(prompt)
            if not question:
                continue

            answer = response if response else completion
            if not answer or not isinstance(answer, str):
                continue

            # Create sample
            sample = {
                "domain": domain,
                "question": question,
                "answer": answer.strip(),
                "data_source": data_source,
                "ability": ability
            }

            collected_samples[domain].append(sample)

        # Report results
        print("\n" + "=" * 60)
        print("EXTRACTION RESULTS")
        print("=" * 60)

        total_samples = 0
        all_extracted = []

        for domain in ["math", "code"]:
            samples = collected_samples[domain]
            print(f"{domain.capitalize()} samples: {len(samples)}")
            total_samples += len(samples)
            all_extracted.extend(samples)

        # Save results
        if total_samples > 0:
            output_path = "data/guru_smart_extracted.jsonl"
            os.makedirs("data", exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                for sample in all_extracted:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            print(f"\n✓ Saved {total_samples} samples to {output_path}")

            # Show samples
            for domain in ["math", "code"]:
                samples = collected_samples[domain]
                if samples:
                    print(f"\n[{domain.upper()} SAMPLE]")
                    sample = samples[0]
                    print(f"Source: {sample['data_source']}")
                    print(f"Question: {sample['question'][:150]}...")
                    print(f"Answer: {sample['answer'][:150]}...")
        else:
            print("\n✗ No samples extracted!")

    except Exception as e:
        print(f"\nError during extraction: {e}")


def extract_user_content_from_chat(chat_messages: List[Dict]) -> Optional[str]:
    """
    Extract user content from OpenAI chat format messages.
    """
    if not isinstance(chat_messages, list):
        return None

    for message in chat_messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def main():
    print("=" * 60)
    print("XRouter Guru Dataset Smart Extraction Tool")
    print("=" * 60)

    # Get dataset info first
    info = get_dataset_info()

    # Extract samples smartly
    extract_samples_smart()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()