from copy import deepcopy


def format_task(task, params):
    task = deepcopy(task)
    for key, value in task.items():
        if isinstance(value, str):
            task[key] = value.format(**params)
        elif isinstance(value, list):
            task[key] = [format_task(v, params) for v in value]
        elif isinstance(value, dict):
            task[key] = format_task(value, params)
    return task

def format_long_task(task, params_list: list):
    task = deepcopy(task)
    task["system"] = task["system"].format(**params_list[0])
    message_template = task["messages"][0]
    task["messages"] = []
    for params in params_list:
        task["messages"].append(format_task(message_template, params))
    return task