import numpy as np

LEFT_EYE = np.arange(36, 42)
RIGHT_EYE = np.arange(42, 48)
EYE = np.concatenate((LEFT_EYE, RIGHT_EYE))

MOUTH = np.arange(48, 60)

EXTERIOR = np.arange(0, 27)

CENTER = 33
ALL_EXCEPT_CENTER = np.setdiff1d(np.arange(60), np.array([CENTER]))

ALL_EXCEPT_EXT = np.setdiff1d(np.arange(60), EXTERIOR)
MOUTH_EYE = np.concatenate([EYE, MOUTH])