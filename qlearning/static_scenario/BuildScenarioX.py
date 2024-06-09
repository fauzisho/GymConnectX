from itertools import product

class BuildScenarioX:
    def __init__(self, list_action, scenario_start_step_default, connected_x):
        self.list_action = list_action
        self.scenario_start_step_default = scenario_start_step_default
        self.connected_x = connected_x

    def generate_permutations(self):
        combos = list(product(self.list_action, repeat=self.scenario_start_step_default))
        valid_combos = []

        for combo in combos:
            count_dict = {action: 0 for action in self.list_action}
            valid = True

            for action in combo:
                count_dict[action] += 1
                if count_dict[action] > self.connected_x:
                    valid = False
                    break

            if valid:
                valid_combos.append(combo)

        return valid_combos

    def check_combos(self, combos):
        for combo in combos:
            count_dict = {action: 0 for action in self.list_action}

            for action in combo:
                count_dict[action] += 1
                if count_dict[action] > self.connected_x:
                    return True, combo

        return False, None