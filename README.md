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

![Image](https://github.com/user-attachments/assets/8a22f19a-8b60-4c67-8fbb-bdcc2767d774)

* **Flappy Bird**

![Image](https://github.com/user-attachments/assets/cef3fdc6-0176-4485-80eb-418601f0ec76)


The visual presentation might seem quite minimalistic, but more focus was put into implementing agent's behavior.


## Implemented Reinforcement agents

So far the only RL algorithm I've tried for the games is Actor-Critic.
