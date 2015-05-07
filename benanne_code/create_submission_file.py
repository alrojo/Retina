import numpy as np
import glob
import os
import sys
import pandas as pd


if len(sys.argv) != 2:
    sys.exit("Usage: python create_submission_file.py <predictions_path>")

predictions_path = sys.argv[1]
predictions = np.load(predictions_path).ravel().astype('int32')

paths = glob.glob("data/test/*.jpeg")
paths.sort()

names = [os.path.splitext(os.path.basename(path))[0] for path in paths]

target_path = os.path.join("submissions", os.path.basename(predictions_path).replace(".npy", ".csv"))

print "Saving to %s" % target_path

df = pd.DataFrame({ 'image': names, 'level': predictions })
df.to_csv(target_path, header=True, index=False)
