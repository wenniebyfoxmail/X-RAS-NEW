import numpy as np
import json


data = np.load("data/fe_sent_phasefield.npz", allow_pickle=True)
print(data.files)


summary = {}

# for key in data.files:
#     arr = data[key]
#     summary[key] = {
#         "shape": arr.shape,
#         "dtype": str(arr.dtype),
#         "min": float(np.nanmin(arr)),
#         "max": float(np.nanmax(arr)),
#         "mean": float(np.nanmean(arr)),
#         "std": float(np.nanstd(arr)),
#         "sample": arr.flatten()[:10].tolist()  # 只取前10个数
#     }
#
# with open("npz_summary.json", "w") as f:
#     json.dump(summary, f, indent=4)
#
# print("✔ Saved npz_summary.json")



keys_to_export = ["d", "u", "reactions"]  # 你需要的变量名

partial = {k: data[k].tolist() for k in keys_to_export if k in data}

with open("npz_partial.json", "w") as f:
    json.dump(partial, f, indent=4)

print("✔ Saved npz_partial.json")

with open("npz_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("✔ Saved npz_summary.json")