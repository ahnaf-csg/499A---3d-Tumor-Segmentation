import sys, subprocess, json
import numpy as np, pandas as pd
print(subprocess.run(['git','-C','/content/499A','pull'], capture_output=True, text=True).stdout)
for k in [x for x in list(sys.modules) if x.startswith('glioseg')]:
    del sys.modules[k]

from glioseg.metrics import halo_analysis
from glioseg.calibrate import THRESHOLDS

# pc_test comes from sweep_streaming; if the session dropped, re-run that cell
h = halo_analysis(pc_test, threshold=0.5, region='ET', thresholds=THRESHOLDS)

print(f"n = {h['n']}")
print(f"fitted halo thickness : {h['halo_thickness_mm_mean']:.2f} mm "
      f"(median {h['halo_thickness_mm_median']:.2f}, SD {h['halo_thickness_mm_sd']:.2f})")
print(f"log-ratio vs log-volume r = {h['log_ratio_vs_log_volume_r']}")
print(f"\n{h['verdict']}\n")
display(pd.DataFrame(h['by_volume']))

PROJ = '/content/drive/MyDrive/Colab Notebooks/499A'
json.dump(h, open(f'{PROJ}/results/halo_analysis.json','w'), indent=2, default=str)
