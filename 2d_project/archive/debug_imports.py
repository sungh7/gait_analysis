print("1. Importing numpy...", flush=True)
import numpy as np
print("2. Importing matplotlib...", flush=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print("3. Importing scipy...", flush=True)
from scipy.signal import resample
print("4. Importing fastdtw...", flush=True)
from fastdtw import fastdtw
print("5. Importing sagittal_extractor_2d...", flush=True)
from sagittal_extractor_2d import MediaPipeSagittalExtractor
print("6. Importing self_driven_segmentation...", flush=True)
from self_driven_segmentation import derive_self_template
print("7. Instantiating Extractor...", flush=True)
extractor = MediaPipeSagittalExtractor()
print("Extractor instantiated.", flush=True)
print("Done.", flush=True)
