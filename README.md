# Adaptive Prisoner's Dilemma Learning Agent

A small Python project exploring how a neural-network policy can adapt during repeated Prisoner's Dilemma games. The main agent uses a one-hidden-layer neural network and a reward-based policy-gradient update. It is an educational experiment, not a claim of state-of-the-art performance.

## Project status

- The main agent and a simple interactive game are implemented.
- The agent uses stochastic action selection: its network output is interpreted as the probability of defecting.
- The agent updates its policy after each round using the received reward relative to a running reward baseline.
- The `Nerual Network Test` directory contains earlier single-layer neural-network experiments. They are kept as prototypes and are not used by the main game.
- No benchmark results are claimed in this README. Performance should be evaluated over repeated games and multiple random seeds before drawing conclusions about which strategy is better.

## How the game works

Actions are encoded as `0 = Cooperate` and `1 = Defect`. The agent's input consists of its own previous nine actions, the opponent's previous nine actions, and a bias input, for 19 input values total.

The main network has:

- 19 input features
- 6 sigmoid hidden units by default
- 1 sigmoid output representing the probability of defection

The agent samples an action from that probability instead of rounding the output. Sampling is important because the reward-based policy-gradient update needs to account for the action the policy selected.

### Learning update

After a round, the agent updates its weights using a one-step REINFORCE-style policy-gradient update. The policy-gradient signal is based on the selected action and the difference between the round's reward and an exponentially updated reward baseline. L2 regularization is applied to the weights.

This is a simple, noisy learning method. It does **not** guarantee that the agent will improve against every opponent, discover an optimal strategy, or maximize long-term reward. Its behavior should be assessed experimentally.

## Payoff matrix

The first action is the human player's action and the second is the agent's action. The values are `(human reward, agent reward)`.

| Human action | Agent action | Human reward | Agent reward |
|---|---|---:|---:|
| Cooperate | Cooperate | 3 | 3 |
| Cooperate | Defect | 0 | 5 |
| Defect | Cooperate | 5 | 0 |
| Defect | Defect | 0 | 0 |

These are the custom payoffs used by this project; they are not the only payoff matrix used in Prisoner's Dilemma examples.

## Requirements

- Python 3.9 or newer recommended
- NumPy

Install NumPy:

```bash
python -m pip install numpy
```

## Run the interactive game

Because the main files are inside a directory containing a space, run:

```bash
cd "Game Theory"
python Game.py
```

Enter `C` to cooperate, `D` to defect, or `Q` to quit. The game displays each round's actions and rewards, plus the cumulative scores.

## Use the agent in another Python script

Run your script from inside the `Game Theory` directory, or add that directory to your Python import path:

```python
from LrBot import AdaptivePrisonerBot

bot = AdaptivePrisonerBot(seed=42)
action = bot.act()  # 0 = cooperate, 1 = defect

# After the opponent's move and payoff are known, calculate the agent reward
# from the payoff matrix before calling learn().
opponent_action = 0
payoffs = {(0, 0): (3, 3), (0, 1): (0, 5),
           (1, 0): (5, 0), (1, 1): (0, 0)}
agent_reward = payoffs[(opponent_action, action)][1]
bot.learn(action, opponent_action, agent_reward)
```

Call `learn()` once per completed round, passing the action actually returned by `act()`, the opponent's action, and the agent's reward for that round.

## Repository structure

```text
GameTheroyBot-main/
├── Game Theory/
│   ├── Game.py          # Interactive game loop
│   ├── LrBot.py         # Main adaptive neural-network agent
│   └── Strategies.py    # Simple opponent strategies for experiments
├── Nerual Network Test/
│   ├── Game-Bot.py      # Earlier single-layer training example
│   ├── MLBot.py         # Earlier single-layer bot prototype
│   └── NettworkClass.py # Earlier neural-network class experiment
└── README.md
```

## Run the tests

From the repository root, run:

```bash
python -m unittest discover -s tests -v
```

The tests check basic action validity, learning/history bookkeeping, invalid action handling, and reset behavior. They do not establish that the agent performs well against particular strategies.

## Limitations and next steps

- The current agent uses a one-step reward signal; it does not explicitly plan over future rounds.
- Policy-gradient updates can have high variance, and the running baseline is only a simple variance-reduction technique.
- The current interactive game is intended for learning and experimentation, not rigorous tournament evaluation.
- A useful next experiment would compare the agent against always-cooperate, always-defect, random, and Tit-for-Tat opponents over many games and random seeds, reporting average rewards and variability.
- The files in `Nerual Network Test` are historical prototypes. The main game does not import them.

## License

No license is specified in this repository yet. Add a license file if you want to clearly state how others may reuse the code.
