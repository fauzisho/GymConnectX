from gymconnectx.envs import ConnectGameEnv
from qlearning.static_scenario.AgentQLearning import AgentQLearning
import pandas as pd

from qlearning.static_scenario.connect4_6x7.scenario_connected4_6x7 import Scenario_Connected4_6x7


def hyperParameter():
    pass


def env_connected4_6x7():
    env = ConnectGameEnv(
        connect=4,
        width=7,
        height=6,
        reward_winner=3,
        reward_loser=-6,
        reward_draw=1,
        reward_hell=-1,
        living_reward=-0.1, )
    return env


def train_agent_env():
    env = env_connected4_6x7()
    scenario = Scenario_Connected4_6x7()
    valid_scenario_train = [scenario.generate_permutations()[0]]

    agent1 = AgentQLearning(epsilon=0.1, role='player_1')
    agent2 = AgentQLearning(epsilon=0.1, role='player_2')

    for n in range(0, len(valid_scenario_train)):
        for game in range(250):
            env.reset()
            scenario_player_1_and_2 = scenario.generate_permutations()[n]
            last_state = ""
            last_action = -1
            last_player = ""
            while not env.is_done:
                try:
                    current_player = 'player_1' if env.get_current_player() == 1 else 'player_2'
                    agent = agent1 if current_player == 'player_1' else agent2
                    if env.current_step < 6:
                        state = str(env.get_player_observations())
                        action = scenario_player_1_and_2[env.current_step]  # based on scenario
                    else:
                        if current_player == 'player_1':
                            state = str(env.get_player_observations())
                            possible_action = env.get_moves()
                            action = agent.choose_action(state, possible_action)
                        else:
                            action = env.get_action_random()

                    next_state, rewards, done, _, info = env.step(action)

                    if env.current_step >= 6 and current_player == 'player_1':
                        agent.update_q_value(state, action, rewards[current_player], next_state)

                    if done:
                        # update agent looser
                        if current_player == 'player_2':
                            agent = agent1
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
        agent1.save_q_table_to_csv(f"1000/case_{n + 1}_player_1_connect4_6x7.csv")
        agent2.save_q_table_to_csv(f"1000/case_{n + 1}_player_2_connect4_6x7.csv")


def test_agent_with_q_table(agent, scenario_player_1_and_2, env):
    env.reset()
    while not env.is_done:
        if env.current_step < 6:
            action = scenario_player_1_and_2[env.current_step]  # based on scenario
        else:
            if env.get_current_player() == 1:
                state = str(env.get_player_observations())
                possible_action = env.get_moves()
                action = agent.choose_action(state, possible_action)
            else:
                action = env.set_players(player_2_mode='random')
                # action = env.set_players(player_2_mode='human_gui')

        next_state, rewards, done, _, info = env.step(action)

        if done:
            return env.get_game_status_player(), str(next_state)
        else:
            env.current_step += 1


def test_with_valid_scenario():
    results = []
    lose_details = []

    valid_scenario_train = Scenario_Connected4_6x7().generate_permutations()
    for i in range(len(valid_scenario_train)):
        win = 0
        draw = 0
        lose = 0

        for n in range(25):
            # q_table = AgentQLearning().load_q_table_from_csv(f'case_{i + 1}_player_1_connect4_4x4.csv')
            q_table = AgentQLearning().load_q_table_from_csv(f'case_117390_player_1_connect4_6x7.csv')
            agent = AgentQLearning(epsilon=0.0, role='player_1', q_table=q_table)
            status, obs_lose = test_agent_with_q_table(
                agent=agent,
                scenario_player_1_and_2=valid_scenario_train[i],
                env=env_connected4_6x7())

            if status == -1:
                draw += 1
            elif status == 1:
                win += 1
            else:
                lose += 1
                lose_details.append({'scenario': valid_scenario_train[i], 'final_lose_state': obs_lose})

        results.append({'scenario': valid_scenario_train[i], 'win': win, 'draw': draw, 'lose': lose})
        print(f'win = {win}, draw = {draw}, lose = {lose}')

    # Saving results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('results_summary.csv', index=False)

    # Saving lose details to another CSV
    lose_details_df = pd.DataFrame(lose_details)
    lose_details_df.to_csv('lose_details.csv', index=False)


def play_with_q_table(scenario=None, file_name=""):
    env = env_connected4_6x7()
    q_table = AgentQLearning().load_q_table_from_csv(file_name)
    agent = AgentQLearning(epsilon=0.0, role='player_1', q_table=q_table)
    env.reset()

    while not env.is_done:
        if scenario == None:
            if env.get_current_player() == 1:
                state = str(env.get_player_observations())
                possible_action = env.get_moves()
                move = agent.choose_action(state, possible_action)
            else:
                move = env.set_players(player_2_mode='human_gui')
        else:
            if env.current_step < 6:
                move = scenario[env.current_step]
            else:
                if env.get_current_player() == 1:
                    state = str(env.get_player_observations())
                    possible_action = env.get_moves()
                    move = agent.choose_action(state, possible_action)
                else:
                    move = env.set_players(player_2_mode='human_gui')

        next_state, rewards, done, _, info = env.step(move)
        env.render('terminal_display')

        if done:
            print(env.get_game_status())
            break
        else:
            env.current_step += 1


if __name__ == "__main__":
    # train_agent_env()
    # test_with_valid_scenario()
    valid_scenario = Scenario_Connected4_6x7().generate_permutations()[0]
    play_with_q_table(
        scenario=valid_scenario,
        file_name="case_1_player_1_connect4_6x7.csv")
