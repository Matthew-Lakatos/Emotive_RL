import random

def run():
    alignment = 0
    human_emotion = 0.5  # simulated tone
    agent_emotion = 0.5

    for iteration in range(50):
        feedback = random.choice(['positive', 'negative'])

        if feedback == 'positive':
            human_emotion += 0.1
            alignment += 2
        else:
            human_emotion -= 0.1
            alignment -= 1

        if abs(human_emotion - agent_emotion) < 0.2:
            alignment += 1

        agent_emotion += (human_emotion - agent_emotion) * 0.1
        agent_emotion += random.uniform(-0.05, 0.05)
        agent_emotion = max(0, min(agent_emotion, 1))

    return min(max(alignment * 2, 0), 100)
