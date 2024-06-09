from qlearning.static_scenario.BuildScenarioX import BuildScenarioX


class ScenarioBase:
    def __init__(self, list_action, scenario_start_step_default, connected_x):
        self.scenario = BuildScenarioX(list_action, scenario_start_step_default, connected_x)

    def generate_permutations(self):
        return self.scenario.generate_permutations()

    def check_combos(self, combos):
        return self.scenario.check_combos(combos)
