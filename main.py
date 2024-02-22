import boto3
import os
import json
import time
import multiprocessing
import asyncio
from collections import defaultdict

client = boto3.client("s3")


async def load_async(cl, path):
    obj = cl.get_object(
        Bucket=os.environ["MODEL_STORE_AWS_BUCKET"],
        Key=path,
    )
    body = obj["Body"].read()
    return json.loads(body)


async def load_all_async(cl, paths):
    coros = [load_async(cl, p) for p in paths]
    return await asyncio.gather(*coros)


def load_parallel(path):
    obj = client.get_object(
        Bucket=os.environ["MODEL_STORE_AWS_BUCKET"],
        Key=path,
    )
    body = obj["Body"].read()
    return json.loads(body)


def load_linear(cl, path):
    # print(f"Linear: {path}")
    bk = os.environ["MODEL_STORE_AWS_BUCKET"]
    obj = cl.get_object(
        Bucket=bk,
        Key=path,
    )
    body = obj["Body"].read()
    return json.loads(body)


if __name__ == '__main__': 
    prefix = os.path.join(
        "example-by-ml-library",
        "operatorai-model-store",
        "domains",
    )

    client = boto3.client("s3")
    objects = client.list_objects_v2(
        Bucket=os.environ["MODEL_STORE_AWS_BUCKET"],
        Prefix=prefix,
    )

    object_paths = []
    for version in objects.get("Contents", []):
        object_path = version["Key"]
        if not object_path.endswith(".json"):
            print("Skipping non-json file: %s", object_path)
            continue
        if os.path.split(object_path)[0] != prefix:
            # We don't want to read files in a sub-prefix
            print("Skipping file in sub-prefix: %s", object_path)
            continue
        object_paths.append(object_path)

    # Scale it up
    summary = defaultdict(int)
    for scale in range(1, 100, 2):
        object_paths = [object_paths[0]] * scale
        start = time.time()
        results = []
        for p in object_paths:
            results.append(load_linear(client, p))
        end = time.time()
        linear = end-start
        assert len(results) == scale
        winner = linear
        strategy = "linear"
        
        start = time.time()
        num_processes = multiprocessing.cpu_count()-1
        with multiprocessing.Pool(num_processes) as pool:
            procs = pool.map(load_parallel, object_paths)
            # results = procs.get()
        end = time.time()
        parallel = end-start
        assert len(results) == scale
        if parallel < winner:
            winner = parallel
            strategy = "parallel"
        parallel = linear - parallel

        start = time.time()
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(load_all_async(client, object_paths))
        end = time.time()
        asynct = end-start
        assert len(results) == scale
        if asynct < winner:
            winner = asynct
            strategy = "async"
        asynct = linear - asynct

        # improvement = linear - parallel
        summary[strategy] += 1
        print(f"{scale}\tLinear: {linear:.4f}\tParallel: {parallel:.2f}\tAsync: {asynct:.2f} (winner={strategy}, {winner:.2f})")
    print(summary)
