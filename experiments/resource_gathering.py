import random

def run():
    mood = 0.5  # baseline mood
    resources = 0

    for step in range(100):
        storm = random.random() < 0.1
        threat = random.random() < 0.1

        if storm or threat:
            mood -= 0.1

        take_risk = mood > 0.4

        if take_risk:
            resources += 1
        else:
            resources += 0.5

        mood += random.uniform(-0.05, 0.05)
        mood = max(0, min(mood, 1))

    return min(max(int(resources), 0), 100)
