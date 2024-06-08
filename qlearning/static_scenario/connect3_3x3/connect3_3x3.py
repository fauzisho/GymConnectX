from gymconnectx.envs import ConnectGameEnv
from qlearning.static_scenario.AgentQLearning import AgentQLearning
from qlearning.static_scenario.connect3_3x3.scenario_3x3 import Scenario_3x3


def hyperParameter():
    pass


def env_3x3():
    env = ConnectGameEnv(
        connect=3,
        width=3,
        height=3,
        reward_winner=3,
        reward_loser=-3,
        reward_draw=1,
        living_reward=-0.1, )
    return env


def train_agent_env():
    env = env_3x3()
    scenario = Scenario_3x3()
    valid_scenario_train = [scenario.generate_permutations()[0]]
    for n in range(0, len(valid_scenario_train)):
        agent = AgentQLearning(epsilon=0.1)
        for game in range(5000):
            env.reset()
            scenario_player_1_and_2 = scenario.generate_permutations()[n]
            state = ""
            action = -1

            player_do_action = ""

            last_state = ""
            last_action = -1
            last_player = ""

            while not env.is_done:
                try:
                    if env.current_step < 4:
                        state = str(env.get_player_observations())
                        action = scenario_player_1_and_2[env.current_step]  # based on scenario
                    else:
                        state = str(env.get_player_observations())
                        possible_action = env.get_moves()
                        move = agent.choose_action(state, possible_action)
                        action = move

                    if env.get_current_player() != 1:
                        player_do_action = "player_1"
                    else:
                        player_do_action = "player_2"

                    next_state, rewards, done, _, info = env.step(action)

                    env.render(mode='terminal_display')
                    if env.current_step >= 4:
                        agent.update_q_value(state, action, rewards[player_do_action], next_state)

                    if done:
                        # update agent looser
                        agent.update_q_value(last_state, last_action, rewards[last_player], next_state)
                        print("------------------------")
                        print(env.get_game_status())
                        print(f'game: {game}, action: {action}, reward: {rewards}')
                        break
                    else:
                        last_player = player_do_action
                        last_state = state
                        last_action = action
                        env.current_step += 1

                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    break
        agent.save_q_table_to_csv(f"case_{n + 1}_connect3_3x3.csv")


if __name__ == "__main__":
    # train_agent_env()
    valid_scenario_train = Scenario_3x3().generate_permutations()[0]
    agent = AgentQLearning(epsilon=0)
    agent.test_agent_with_q_table(scenario_player_1_and_2=valid_scenario_train,
                                  env=env_3x3(),
                                  file_name="case_1_connect3_3x3.csv")
