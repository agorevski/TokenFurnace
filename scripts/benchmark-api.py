#!/usr/bin/env python3

import argparse
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Write a Python quicksort implementation.")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    usage = result.get("usage", {})
    output_tokens = usage.get("completion_tokens", 0)
    print(json.dumps({
        "elapsed_seconds": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / elapsed if output_tokens else None,
        "finish_reason": result["choices"][0].get("finish_reason"),
    }, indent=2))


if __name__ == "__main__":
    main()
