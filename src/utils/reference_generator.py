from typing import List


def generate_reference(target_temps: List[float], target_times: List[int]):
    assert len(target_times) + 1 == len(target_temps)

    entire_time = sum(target_times)

    ref_temps = []

    for step_idx in range(len(target_times)):
        for time_idx in range(target_times[step_idx]):
            ref_temps.append(target_temps[step_idx] + time_idx * (target_temps[step_idx+1] - target_temps[step_idx]) / (target_times[step_idx]))
    return ref_temps