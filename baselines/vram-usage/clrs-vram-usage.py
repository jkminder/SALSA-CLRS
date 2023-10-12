import math
import csv

import sys
import subprocess
import argparse

def main(processor_type):
    results = {}

    for p in range(0, 15):
        scale = 2 ** p
        n = 2 * scale
        p = math.log(n) / n
        
        try:
            output = subprocess.check_output([sys.executable, "baselines/vram-usage/_clrs_run.py", f"-n", str(n), "--batch_size", "1", "--processor_type", processor_type ])
        except subprocess.CalledProcessError as e:
            results[n] = "OOM"
            print(f"n: {n}, p: {p}, usage: OOM")
            break
        output = output.decode("utf-8")

        results[n] = int(output.replace("\n", ""))

        # get the vram usage 
        print(f"n: {n}, p: {p}, usage: {results[n]/(1024 * 1024)} MiB")

    print(results)

    # save as csv
    with open(f"vram-usage-clrs-{processor_type}.csv", "w") as f:
        writer = csv.DictWriter(f, results.keys())
        writer.writeheader()
        writer.writerow(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor_type", type=str, default="triplet_mpnn")
    args = parser.parse_args()


    main(args.processor_type)