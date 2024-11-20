import warnings
import gym
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import argparse
import numpy as np
from Agent import Agent


ENVIRONMENT = "CarRacing-v1"


def run_game(agent, num_games, max_timesteps, no_gui=False, deploy=False, seed=None):

    # Use a random seed if passed as argument
    if seed:
        np.random.seed(seed)
    
    # Create the environment
    env = gym.make(ENVIRONMENT)

    # Run the game num_games times
    for i_episode in range(num_games):

        # Reset the environment for each game
        observation = env.reset()
        reward = 0

        # Inform the agent that a new game is starting
        agent.start_new_game()

        # Iterate over time
        for t in range(max_timesteps):

            # Draw the current environment if GUI enabled
            if not no_gui:
                env.render()

            # Give agent the current observation and reward
            # Get the agent's next action
            action = agent.act(observation, reward)

            # Execute the action and get results
            observation, reward, done, info = env.step(action)
           
            # End game early if task complete
            if done:
                print("Game finished after {} timesteps".format(t+1))
                break

        # Print final reward
        if not done:
            print("Game timed out after {} timesteps".format(t+1))
        print("Final reward: {}".format(reward))

    # Clear game
    env.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run reinforcement learning game")
    parser.add_argument("num_games", type=int, help="Number of times to run the game (int)")
    parser.add_argument("max_timesteps", type=int, help="Maximum number of steps to run each game (int)")
    parser.add_argument("-i", "--input", type=str, help="Identifier for loading trained model (string)")
    parser.add_argument("-o", "--output", type=str, help="Identifier for saving trained model (string)")
    parser.add_argument("-g", "--nogui", help="Run game without graphics for faster training", action="store_true")
    parser.add_argument("-d", "--deploy", help="Run in deployment mode", action="store_true")
    parser.add_argument("-s", "--seed", type=int, help="Random seed for track generation (int)")
    args = parser.parse_args()

    # Create the agent
    agent = Agent(args.input)

    # Run the simulation
    run_game(agent, args.num_games, args.max_timesteps, args.nogui, args.deploy, args.seed)

    # Save the agent
    if args.output:
        agent.save(args.output)


if __name__ == "__main__":
    main()
