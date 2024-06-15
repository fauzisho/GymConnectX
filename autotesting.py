from gymconnectx.envs import ConnectGameEnv

winner_reward = 3
loser_reward = -3
living_reward = -0.01
draw_reward = 0.5
hell_reward = -0.9
prob_hell_reward = -0.75
prob_win_reward = 0.75


def run_game_env():
    env = ConnectGameEnv(
        connect=3,
        width=3,
        height=3,
        reward_winner=winner_reward,
        reward_loser=loser_reward,
        reward_living=living_reward,
        reward_draw=draw_reward,
        reward_hell=hell_reward,
        reward_hell_prob=prob_hell_reward,
        reward_win_prob=prob_win_reward,
        max_steps=100,
        delay=100,
        square_size=100,
        avatar_player_1='img_dog.png',
        avatar_player_2='img_cat.png')

    env.reset()

    action_list = [[2, 2, 1, 2, 0]]
    values_list = [[living_reward, living_reward, prob_hell_reward, hell_reward, winner_reward]]
    for k in range(len(action_list)):
        actions = action_list[k]
        values = values_list[k]

        for i in range(len(actions)):
            move = actions[i]
            value = values[i]

            if env.get_current_player() == 1:
                player = 'player_1'
            else:
                player = "player_2"

            try:
                observations, rewards, done, _, info = env.step(move)
                if rewards[player] == value:
                    print(f'pass')
                else:
                    print(f'action {move} get reward {rewards[player]} == {value}')
                    print("failed")
                if done:
                    print(env.get_game_status())
                    break
                else:
                    env.current_step += 1

            except Exception as e:
                print(f"An error occurred: {str(e)}")
                break


if __name__ == "__main__":
    run_game_env()
