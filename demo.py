import gymconnectx
import gym

if __name__ == "__main__":
    env = gym('gymconnectx/ConnectGameEnv')
    env.reset()

    while not env.is_done and env.current_step < env.max_steps:
        try:
            move = env.set_players(player_1_mode='random', player_2_mode='random')
            observations, rewards, done, info = env.step(move)
            env.render(mode='terminal_display')
            env.render(mode='gui_update_display')
            print(f"Step: {env.current_step}, "
                  f"Move: {move}, "
                  f"Rewards: {rewards}, "
                  f"Done: {done}, "
                  f"Info: {info}")
            print(env.get_game_status())
            if done:
                break
            else:
                # Increment the step counter if the game is not finished.
                env.current_step += 1

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break