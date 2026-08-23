import matplotlib.pyplot as plt


class Task:
    def __init__(self, task_name=None, template=None, mappings=[], final_mapping=None):
        self.task_name = task_name
        self.template = template
        self.mappings = mappings
        self.final_mapping = final_mapping


log_file = open("./log_file.txt", "r", encoding="utf-8")
log_file_lines = log_file.read().split("\n")

all_tasks = []
task = Task()
for line in log_file_lines:
    if line.startswith("selected task"):
        task.task_name = line.split(":")[1].strip()
    elif line.startswith("selected template"):
        task.template = int(line.split(":")[1].strip())
    elif line.startswith("mapping") and "dev" not in line:
        task.mappings.append(float(line.split(":")[1].strip()))
    elif line.startswith("after train"):
        task.final_mapping = int(line.split(" ")[3].strip())
        all_tasks.append(task)
        task = Task("",-1,[], -1)

tasks_patterns = {
        "parsinlu-food-sentiment": [1, 2, 3],
        "parsinlu-movie-sentiment": [1, 2, 3],
        "parsinlu-nli": [1, 2, 3],
        "digikala-tc": [1, 2]
    }



for task_name in tasks_patterns.keys():
    fig, axes = plt.subplots(1, len(tasks_patterns[task_name]), figsize=(18, 15))
    fig.suptitle(task_name)
    for counter, template_no in enumerate(tasks_patterns[task_name]):
        info = [t for t in all_tasks if t.task_name == task_name and t.template == template_no]
        all_mapping = [t.mappings for t in info]
        for mapping_id, values in enumerate(zip(*all_mapping)):
            axes[counter].plot(range(len(values)), values, linewidth=2, label=f"Mapping {mapping_id}")

        axes[counter].set_title(f"template {template_no}")
        #axes[counter].legend()
        axes[counter].legend(
            ncol=4,
            fontsize=8,
            bbox_to_anchor=(0.5, -0.2),
            loc="upper center"
        )

    plt.tight_layout()
    plt.show()
