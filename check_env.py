from datacenter import DataCenterEnv
import numpy as np

env = DataCenterEnv("data/datacenter.csv")
s = env.reset()
print("State:", s)
print("State min/max:", float(np.min(s)), float(np.max(s)))
