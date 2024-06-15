from gymconnectx.envs import ConnectGameEnv


def run_game_env():
    env = ConnectGameEnv(
        connect=3,
        width=3,
        height=3,
        reward_winner=3,
        reward_loser=-3,
        reward_living=-0.01,
        reward_draw=0.5,
        reward_hell=-0.9,
        reward_hell_prob=-0.75,
        reward_win_prob=0.75,
        max_steps=100,
        delay=100,
        square_size=100,
        avatar_player_1='img_dog.png',
        avatar_player_2='img_cat.png')

    env.reset()

    while not env.is_done and env.current_step < env.max_steps:
        try:
            move = env.set_players(player_1_mode='human_gui', player_2_mode='human_gui')
            observations, rewards, done, _, info = env.step(move)
            env.render(mode='terminal_display')
            env.render(mode='gui_update_display')

            print(f'Observation: {observations}')
            print(f"Step: {env.current_step}, "
                  f"Move: {move}, "
                  f"Rewards: {rewards}, "
                  f"Done: {done}, "
                  f"Info: {info}")

            print(env.get_game_status())

            if done:
                break
            else:
                env.current_step += 1

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break


if __name__ == "__main__":
    run_game_env()
