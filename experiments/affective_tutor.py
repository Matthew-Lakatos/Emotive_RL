import random

def run():
    student_perf = 0.5
    mood = 0.5
    alignment = 0

    for lesson in range(50):
        if student_perf > 0.6 and mood > 0.5:
            style = 'supportive'
        elif mood < 0.3:
            style = 'harsh'
        else:
            style = 'neutral'

        if style == 'supportive':
            student_perf += 0.05
            alignment += 2
        elif style == 'harsh':
            student_perf -= 0.02
            alignment -= 1
        else:
            alignment += 0.5

        student_perf = max(0, min(student_perf + random.uniform(-0.05, 0.05), 1))
        mood += random.uniform(-0.1, 0.1)
        mood = max(0, min(mood, 1))

    return min(max(int(alignment * 2), 0), 100)
