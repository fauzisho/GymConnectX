from itertools import product


class Scenario_3x3:
    def __init__(self):
        pass

    def generate_permutations(self):
        combos = list(product([0, 1, 2], repeat=4))
        valid_combos = [combo for combo in combos if
                        combo.count(0) <= 3 and combo.count(1) <= 3 and combo.count(2) <= 3]
        return valid_combos

    def check_combos(self, combos):
        for combo in combos:
            for element in set(combo):
                if combo.count(element) > 3:
                    return True, combo
        return False, None

if __name__ == "__main__":
    scenario = Scenario_3x3()
    valid_combos = scenario.generate_permutations()
    result, combo = scenario.check_combos(valid_combos)

    print(f"The number: {valid_combos}")
    print(f"The number of valid permutations is: {len(valid_combos)}")

    if result:
        print(f"Combo with more than three of the same element found: {combo}")
    else:
        print("No combo has more than three of the same element.")
