from itertools import product

from qlearning.static_scenario.ScenarioBase import ScenarioBase


# width 7, height 6
class Scenario_Connected4_6x7(ScenarioBase):
    def __init__(self):
        super().__init__(list_action=[0, 1, 2, 3, 4, 5, 6],
                         scenario_start_step_default=6,
                         connected_x=4)


if __name__ == "__main__":
    scenario = Scenario_Connected4_6x7()

    valid_combos = scenario.generate_permutations()
    result, combo = scenario.check_combos(valid_combos)

    print(f"The number: {valid_combos}")
    print(f"The number of valid permutations is: {len(valid_combos)}")

    if result:
        print(f"Combo with more than {4} of the same element found: {combo}")
    else:
        print(f"No combo has more than {4} of the same element.")
