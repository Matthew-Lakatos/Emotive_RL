import random

def run():
    morale = 0.7
    ops_success = 0

    for day in range(100):
        isolation = random.random() < 0.1
        if isolation:
            morale -= 0.1

        task_success = morale > 0.3
        if task_success:
            ops_success += 1

        morale += random.uniform(-0.05, 0.05)
        morale = max(0, min(morale, 1))

    return min(ops_success, 100)
