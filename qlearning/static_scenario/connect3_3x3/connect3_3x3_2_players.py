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
        reward_hell=- 1,
        living_reward=-0.1, )
    return env


def train_agent_env():
    env = env_3x3()
    scenario = Scenario_3x3()
    valid_scenario_train = [scenario.generate_permutations()[0]]

    agent1 = AgentQLearning(epsilon=0.1, role='player_1')
    agent2 = AgentQLearning(epsilon=0.1, role='player_2')

    for n in range(0, len(valid_scenario_train)):
        for game in range(5000):
            env.reset()
            scenario_player_1_and_2 = scenario.generate_permutations()[n]
            last_state = ""
            last_action = -1
            last_player = ""
            while not env.is_done:
                try:
                    current_player = 'player_1' if env.get_current_player() == 1 else 'player_2'
                    agent = agent1 if current_player == 'player_1' else agent2
                    if env.current_step < 4:
                        state = str(env.get_player_observations())
                        action = scenario_player_1_and_2[env.current_step]  # based on scenario
                    else:
                        state = str(env.get_player_observations())
                        possible_action = env.get_moves()
                        action = agent.choose_action(state, possible_action)

                    print(f'player :{current_player} : action {action}')
                    next_state, rewards, done, _, info = env.step(action)
                    print(f'game: {game}, action: {action}, reward: {rewards}')

                    env.render(mode='terminal_display')
                    if env.current_step >= 4:
                        agent.update_q_value(state, action, rewards[current_player], next_state)

                    if done:
                        # update agent looser
                        current_player = 'player_1' if env.get_current_player() == 1 else 'player_2'
                        agent = agent1 if current_player == 'player_1' else agent2
                        agent.update_q_value(last_state, last_action, rewards[last_player], next_state)
                        print("------------------------")
                        print(env.get_game_status())
                        print(f'game: {game}, action: {action}, reward: {rewards}')
                        break
                    else:
                        last_player = current_player
                        last_state = state
                        last_action = action
                        env.current_step += 1

                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    break
        agent1.save_q_table_to_csv(f"case_{n + 1}_player_1_connect3_3x3.csv")
        agent2.save_q_table_to_csv(f"case_{n + 1}_player_2_connect3_3x3.csv")


if __name__ == "__main__":
    # train_agent_env()
    valid_scenario_train = [Scenario_3x3().generate_permutations()[0]]
    for i in range(0, len(valid_scenario_train)):
        win = 0
        draw = 0
        lose = 0
        for n in range(0, 100):
            # player 2
            agent = AgentQLearning(epsilon=0)
            status = agent.test_agent_with_q_table(scenario_player_1_and_2=valid_scenario_train[i],
                                                   env=env_3x3(),
                                                   file_name=f'case_{i + 1}_player_1_connect3_3x3.csv')
            if status == -1:
                draw += 1
            elif status == 1:
                lose += 1
            else:
                win += 1
        print(f'win = {win}, draw = {draw}, lose = {lose}')
