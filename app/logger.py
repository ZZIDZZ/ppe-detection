from time import time
prev_detected = {}
now_detected = {}
def log(det, names, detected, prev_detected):
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