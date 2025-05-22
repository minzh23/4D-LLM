from datasets import load_dataset

ds = load_dataset("nyu-visionx/VSI-Bench")
ds.save_to_disk("VSI-Bench")