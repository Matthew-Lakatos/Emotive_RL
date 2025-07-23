import random

def run():
    mood = 0.5
    boldness = 0.5
    discoveries = 0

    for step in range(100):
        stimulus = random.choice(['pleasant', 'threatening', 'neutral'])

        if stimulus == 'pleasant':
            mood += 0.1
        elif stimulus == 'threatening':
            mood -= 0.1

        boldness = mood

        if boldness > 0.5:
            discoveries += 1
        else:
            discoveries += 0.5

        mood += random.uniform(-0.05, 0.05)
        mood = max(0, min(mood, 1))

    return min(max(int(discoveries), 0), 100)
