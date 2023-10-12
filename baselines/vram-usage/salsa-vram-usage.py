import sys
sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("./baselines") 
from salsaclrs import load_dataset, SALSACLRSDataset, SALSACLRSDataLoader
from core.config import load_cfg
from core.module import SALSACLRSModel
import math
import threading as th
import csv
import torch
import argparse
from core.config import load_cfg

def main(DATA_DIR, cfg, checkpoint, num_samples=10):
    ds = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="bfs", num_samples=1, graph_generator="er", graph_generator_kwargs={"n": [16], "p_range": (0.1,0.1)}, hints=True)

    if checkpoint is not None:
        model = SALSACLRSModel.load_from_checkpoint(checkpoint).cuda()
    else:
        model = SALSACLRSModel(specs=ds.specs, cfg=cfg).cuda()
    print(model)
    print(model.hparams)
    print(torch.cuda.memory_allocated())
    model.eval()

    results = {}
    for p in range(0, 15):
        torch.cuda.reset_peak_memory_stats()

        scale = 2 ** p
        n = 2 * scale
        p = math.log(n) / n

        ds = SALSACLRSDataset(root=DATA_DIR, split="train", algorithm="bfs", num_samples=num_samples, graph_generator="er", graph_generator_kwargs={"n": [n], "p_range": (p*1, p*3)}, hints=True, max_cores=0)

        dl = SALSACLRSDataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        for batch in dl:    
            model(batch.cuda())
            # remove batch from memory
            batch.detach()
            del batch

        # get the vram usage 
        usage = torch.cuda.max_memory_allocated()

        print(f"n: {n}, p: {p}, usage: {usage/(1024 * 1024)} MiB")

        results[n] = usage

    print(results)

    # save as csv
    with open("vram-usage-salsa.csv", "w") as f:
        writer = csv.DictWriter(f, results.keys())
        writer.writeheader()
        writer.writerow(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--cfg", type=str, default="./baselines/configs/bfs/GIN.yml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load. If not specified, will initiate a new model based on --cfg")
    args = parser.parse_args()

    if args.checkpoint is None:
        cfg = load_cfg(args.cfg)
    else:
        cfg = None
    main(args.data_dir, cfg, args.checkpoint)
