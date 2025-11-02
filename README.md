# RL Framework for Classic Games

A small Python framework for training, and visualizing reinforcement learning agents in classic single player games. This project uses PyTorch for AI agents and Pygame for visualization.

## Setup and Installation

The package manager used for this project is [uv](https://docs.astral.sh/uv/getting-started/installation/).
To create environment and install dependencies,

```bash
git clone https://github.com/mnozkiewicz/RL-Games
cd RL-Games

uv sync
source .venv/bin/activate
```

## Downloading assets and weights

Assets are just images for every game used in pygame.
Weights are the parameters of the rl-agents trained to play the games.

In order to and start pygame gui (instructions for this below), you either need to download assets manually from 
[here](https://drive.google.com/drive/folders/18b3NMpB9F8lXTvMrtFHyzmMCD4hE_d1R?usp=sharing), and replace the assets folder with the downloaded one or use 
the script to do this automatically.

```bash
chmod +x ./scripts/download_assets.sh 
./scripts/download_assets.sh 
```

In order to use the trained AI agent, similarly you can download the weights from [here](https://drive.google.com/drive/folders/1AFCSASPzFUkPk6xseTsKVdwC_a9HkIBj?usp=sharing)
or use a script.

```bash
chmod +x ./scripts/download_weights.sh
./scripts/download_weights.sh
```

## How to Run

In order to check the options for running the game type

```bash
uv run -m src.gui --help
```

You can play the on your own.
```bash
## Playing on your own
uv run -m src.gui --game snake --player human --infinite
```

Or use an rl-agent for this.
```bash
## With --learn argument the agent will learn to play snake during the game visualization
uv run -m src.gui --game snake --player ai --learn --infinite
```


If you want to first train the agent you can use the option to train without visualisation (which slows down the process).
```bash
## Check possible arguments
uv run -m src.training.train --help

## Traing rl-agent
uv run -m src.training.train --game snake --episodes 1000 --max-steps 3000
```

For this particular command the agent weights should be saved under "src/training/SNAKE_controller".
Then you can run.

```bash
uv run -m src.gui --game snake --player ai --pretrained --infinite
```


## Implemented games

So far I've implemented (and trained agent to play them) for two games:

* **Snake**

![Snake_gif](https://github.com/user-attachments/assets/87c7753b-b4f8-481d-a65c-3e2dd2adc7d7)

* **Flappy Bird**

![Flappy_gif](https://github.com/user-attachments/assets/b1f97d8c-2f24-427c-b625-1a6aeff7a549)


## Implemented Reinforcement agents

So far the only RL algorithm I've tried for the games is Actor-Critic.
