from itertools import count
from time import time
import numpy as np

prev_detected = {}
now_detected = {}

def log(det, names, detected, prev_detected):
    #TODO: try sqlite, or json
    s=""
    for c in det[:, -1].unique():
        n = int((det[:, -1] == c).sum())  # detections per class
        if (names[int(c)] not in detected):
            detected.update({names[int(c)] : n})
        if (names[int(c)] not in prev_detected):
            prev_detected.update({names[int(c)] : n})
        if (names[int(c)] not in now_detected):
            now_detected.update({names[int(c)] : n})
        dif = now_detected[names[int(c)]] - prev_detected[names[int(c)]]
        if(dif > 0):
            detected[names[int(c)]] += dif
        # s += f"{n} {names[int(c)]} "
        # print(names[int(c)])
    with open("logs/asdlog.txt", 'a') as f:
        f.write(s + "\n")
        print(detected, int(round(time())))
    return detected,now_detected



def logv2(det, names, detected, prev_detected):
    """
    apa yang berbeda dari log?
    logv2 melakukan group class tensor
    
    """
    s = ""

    obj_seen, count = np.unique(det[:, -1].cpu(), return_counts=True)

    for obj, c in zip(obj_seen, count):
        s += f"{c} {names[int(obj)]} "
    print(s)

    return detected,now_detected  