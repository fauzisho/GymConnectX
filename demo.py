import gymconnectx

def run_game_env():
    env = gymconnectx.gym.make('gymconnectx/ConnectGameEnv',
                               connect=4,
                               width=7,
                               height=6,
                               reward_winner=1,
                               reward_loser=-1,
                               living_reward=0, max_steps=100, delay=100, square_size=100,
                               avatar_player_1='img_cat.png', avatar_player_2='img_cat.png')
    env.reset()

    while not env.is_done and env.current_step < env.max_steps:
        try:
            move = env.set_players(player_1_mode='human_gui', player_2_mode='random')
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
