from itertools import product

from qlearning.static_scenario.ScenarioBase import ScenarioBase


class Scenario_3x3(ScenarioBase):
    def __init__(self):
        super().__init__(list_action=[0, 1, 2],
                         scenario_start_step_default=4,
                         connected_x=3)

class Scenario_3x3_step_2(ScenarioBase):
    def __init__(self):
        super().__init__(list_action=[0, 1, 2],
                         scenario_start_step_default=2,
                         connected_x=3)


if __name__ == "__main__":
    scenario = Scenario_3x3()
    # scenario = Scenario_3x3_step_2()

    valid_combos = scenario.generate_permutations()
    result, combo = scenario.check_combos(valid_combos)

    print(f"The number: {valid_combos}")
    print(f"The number of valid permutations is: {len(valid_combos)}")

    if result:
        print(f"Combo with more than {3} of the same element found: {combo}")
    else:
        print(f"No combo has more than {3} of the same element.")
