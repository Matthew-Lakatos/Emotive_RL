import random

def run():
    score = 0
    harmony = 0
    for step in range(100):
        agent_action = random.choice(['approach', 'avoid', 'cooperate'])
        other_agent_emotion = random.choice(['happy', 'angry', 'neutral'])

        if agent_action == 'cooperate' and other_agent_emotion == 'happy':
            harmony += 2
        elif agent_action == 'avoid' and other_agent_emotion == 'angry':
            harmony += 1
        elif agent_action == 'approach' and other_agent_emotion == 'angry':
            harmony -= 2
        else:
            harmony += 0

        score += harmony

    return min(max(score // 10, 0), 100)
