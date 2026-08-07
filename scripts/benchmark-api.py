#!/usr/bin/env python3

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def one_request(base_url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "******"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    output_tokens = usage.get("completion_tokens", 0) or 0
    return {
        "elapsed_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "end_to_end_prompt_tokens_per_second":
            prompt_tokens / elapsed if prompt_tokens else None,
        "output_tokens_per_second": output_tokens / elapsed if output_tokens else None,
        "finish_reason": result["choices"][0].get("finish_reason"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Write a Python quicksort implementation.")
    parser.add_argument("--prompt-file",
                        help="Read the prompt from a UTF-8 file")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of requests issued in parallel (default 1)")
    parser.add_argument("--requests", type=int, default=None,
                        help="Total requests to issue (default: --concurrency)")
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    concurrency = args.concurrency
    total = args.requests if args.requests is not None else concurrency
    if total < 1:
        parser.error("--requests must be at least 1")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as prompt_file:
            prompt = prompt_file.read()

    # Single-request path keeps the original output shape for compatibility.
    if concurrency == 1 and total == 1:
        print(json.dumps(one_request(args.base_url, args.model, prompt,
                                     args.max_tokens), indent=2))
        return

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one_request, args.base_url, args.model,
                               prompt, args.max_tokens)
                   for _ in range(total)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - started

    prompt_tokens = sum(r["prompt_tokens"] for r in results)
    completion_tokens = sum(r["completion_tokens"] for r in results)
    per_req = [r["output_tokens_per_second"] for r in results
               if r["output_tokens_per_second"] is not None]
    mean_single = sum(per_req) / len(per_req) if per_req else None
    print(json.dumps({
        "concurrency": concurrency,
        "total_requests": total,
        "wall_seconds": wall,
        "total_prompt_tokens": prompt_tokens,
        "total_completion_tokens": completion_tokens,
        "aggregate_end_to_end_prompt_tokens_per_second":
            prompt_tokens / wall if prompt_tokens else None,
        "aggregate_output_tokens_per_second":
            completion_tokens / wall if completion_tokens else None,
        "mean_single_request_tokens_per_second": mean_single,
    }, indent=2))


if __name__ == "__main__":
    main()
