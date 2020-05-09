#!/bin/env python3
# Evan Widloski - 2020-05-09

import numpy as np
import matplotlib.pyplot as plt
from sr549.video import Video

v = Video()


plt.figure()
plt.imshow(v.frames[0])
plt.figure()
plt.imshow(v.frames_clean[0])

print(v.true_drift)
