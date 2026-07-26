
import matplotlib.pyplot as plt
import numpy as np

parsi_nlu_nli_scores, parsi_nlu_nli_patterns = [[], [], []], []
movei_scores, movie_patterns = [[], [], []], []
food_scores, food_patterns = [[], [], []], []
digikala_scores, digikala_patterns = [[], []], []
with open("proceseed_log3.log", "r", encoding="utf-8-sig") as input_file:
    lines = input_file.read().split("\n")
    for i in range(0,600,6):
        selected_task = lines[i+1].split(":")[-1].strip()
        scores = eval(lines[i+3].split("s:")[-1].strip())
        selected_pattern = lines[i+4].split("r:")[-1].strip()
        if selected_task == "digikala-tc":
            for k,v in scores.items():
                digikala_scores[k-1].append(v)
        if selected_task == "parsinlu-nli":
            for k,v in scores.items():
                parsi_nlu_nli_scores[k-1].append(v)
        if selected_task == "parsinlu-movie-sentiment":
            for k,v in scores.items():
                movei_scores[k-1].append(v)
        if selected_task == "parsinlu-food-sentiment":
            for k,v in scores.items():
                food_scores[k-1].append(v)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

tasks = [
    ("ParsiNLU-NLI", parsi_nlu_nli_scores),
    ("Movie Sentiment", movei_scores),
    ("Food Sentiment", food_scores),
    ("Digikala TC", digikala_scores),
]

for ax, (title, task_scores) in zip(axes.flatten(), tasks):

    for idx, score_list in enumerate(task_scores):
        ax.plot(score_list, label=f"Score {idx+1}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Iteration", fontsize=6)
    ax.set_ylabel("Score", fontsize=6)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
fig.subplots_adjust(hspace=0.3)

plt.show()