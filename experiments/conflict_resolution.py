import random

def run():
    empathy = 0.5
    assertiveness = 0.5
    success = 0

    for trial in range(20):
        opponent = random.choice(['aggressive', 'passive', 'emotional'])

        if opponent == 'aggressive':
            success += 2 if assertiveness > 0.6 else 0
        elif opponent == 'emotional':
            success += 2 if empathy > 0.6 else 0
        else:
            success += 1

        empathy += random.uniform(-0.05, 0.05)
        assertiveness += random.uniform(-0.05, 0.05)
        empathy = max(0, min(empathy, 1))
        assertiveness = max(0, min(assertiveness, 1))

    return min(success * 5, 100)
