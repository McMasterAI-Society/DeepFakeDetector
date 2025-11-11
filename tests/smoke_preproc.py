# tests/smoke_preproc.py  (optional helper)
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from datautil.preproc_dataset import PreprocDataset, collate

ds = PreprocDataset(Path("data/sample"))
print("Total samples:", len(ds))
dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
batch = next(iter(dl))

print("vit:", batch["vit"].shape)          # [B,3,S,S]
print("fft:", batch["fft"].shape)          # [B,1,P,P]
print("patches[0]:", batch["patches"][0].shape)  # [N0,3,P,P]
print("labels:", batch["labels"].shape, batch["labels"])
print("meta[0]:", batch["meta"][0])
