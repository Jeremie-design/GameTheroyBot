# Adaptive Neural Network for the Repeated Prisoner's Dilemma

A Python experiment exploring how a small neural network can adapt its behavior in repeated Prisoner's Dilemma games.

The project combines neural-network architecture, reward-driven weight updates, and classical game-theoretic strategies to investigate how an agent's recent interaction history relates to its future decisions.

Rather than relying entirely on predefined behavioral rules, the agent uses a neural network to process previous actions and update its weights after receiving each round's payoff.

The project focuses on understanding learning dynamics and strategic behavior rather than claiming optimal performance.

## Project Overview

The Prisoner's Dilemma is a game-theoretic model in which two players independently choose between cooperating and defecting.

Each action affects both players' rewards:

- **Cooperate (0):** Choose mutual cooperation or accept the possibility of exploitation.
- **Defect (1):** Attempt to maximize individual payoff at the expense of cooperation.

In a repeated game, players can use previous interactions to inform future decisions.

This project investigates whether a neural-network-based agent can adjust its behavior based on patterns in its own actions, its opponent's actions, and the rewards it receives.

## Agent Architecture

The main agent is implemented in `Game Theory/LrBot.py`.

It uses a feed-forward neural network with one hidden layer.

### 1. Input Representation

The agent maintains a sliding history of the previous nine actions from each player.

The input vector contains:

| Component | Size |
|---|---:|
| Agent's previous actions | 9 |
| Opponent's previous actions | 9 |
| Bias input | 1 |
| **Total input features** | **19** |

Actions are numerically encoded:

- `0` = Cooperate
- `1` = Defect

The input vector is constructed as:

\[
x = [a_1,\ldots,a_9,o_1,\ldots,o_9,1]
\]

where:

- \(a_i\) represents one of the agent's previous actions.
- \(o_i\) represents one of the opponent's previous actions.
- The final value is the bias input.

At initialization, both histories are filled with zeros.

As the simulation progresses, the oldest action is removed and the newest action is appended.

### 2. Network Structure

The network contains three computational components:

| Layer | Units | Activation |
|---|---:|---|
| Input | 19 | None |
| Hidden | 6 | Sigmoid |
| Output | 1 | Sigmoid |

The forward pass consists of two transformations.

First, the hidden layer processes the input:

\[
h = \sigma(xW_1)
\]

A bias value is then appended to the hidden representation before calculating the output:

\[
y = \sigma([h,1]W_2)
\]

where:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The weight matrices have the following dimensions:

- \(W_1 \in \mathbb{R}^{19 \times 6}\)
- \(W_2 \in \mathbb{R}^{7 \times 1}\)

The sigmoid output is converted into a binary action using `np.round()`:

- `0` = Cooperate
- `1` = Defect

The output is not sampled as a probability. It is deterministically rounded into an action.

## Reward-Based Learning

After each round, the agent receives a numerical payoff determined by the game's payoff matrix.

The `learn()` method uses the normalized reward as a target for the neural network's output.

### 1. Reward Normalization

The agent's reward is divided by the maximum possible reward:

\[
r_{\text{normalized}} = \frac{r}{5}
\]

This scales the reward to a value between zero and one for the current payoff matrix.

The normalized reward is used as the target value for the network output.

### 2. Output Error

The difference between the normalized reward and the network output determines the output-layer error:

\[
\delta_o =
(r_{\text{normalized}}-y)y(1-y)
\]

The sigmoid derivative is:

\[
\sigma'(y)=y(1-y)
\]

### 3. Backpropagation

The output-layer error is propagated backward through the hidden layer:

\[
\delta_h =
(\delta_o W_{2,h}^{T})\odot h(1-h)
\]

Here, \(W_{2,h}\) refers to the hidden-unit weights in the output layer, excluding its bias weight.

The network then updates both weight matrices using the learning rate and an L2 regularization term.

\[
W_2 \leftarrow W_2+
\alpha(h_b^T\delta_o-\lambda W_2)
\]

\[
W_1 \leftarrow W_1+
\alpha(x^T\delta_h-\lambda W_1)
\]

where:

- \(\alpha\) is the learning rate.
- \(\lambda\) is the L2 regularization coefficient.
- \(h_b\) is the hidden representation with its bias input.
- \(\odot\) represents element-wise multiplication.

### 4. Learning Configuration

The current implementation uses:

| Parameter | Default Value |
|---|---:|
| Memory size | 9 actions per player |
| Hidden units | 6 |
| Learning rate | 0.30 in the simulation |
| L2 regularization | 0.001 |
| Reward normalization | Maximum reward of 5 |

The update is an online, reward-targeted, backpropagation-style learning rule.

**Important distinction:** This is not a REINFORCE or policy-gradient implementation. The network does not sample an action from a probability distribution or calculate a gradient using the log-probability of a selected action.

Instead, its output is trained toward the normalized reward from the completed round and then used to select the next binary action.

The current implementation is an exploratory learning approach, not a demonstrated optimal decision-making algorithm.

## Payoff Matrix

The simulation uses the following payoff matrix.

The first action represents the neural-network agent, and the second represents its opponent.

Each cell contains `(agent payoff, opponent payoff)`.

| Agent Action | Opponent Action | Agent Reward | Opponent Reward |
|---|---|---:|---:|
| Cooperate | Cooperate | 3 | 3 |
| Cooperate | Defect | 0 | 5 |
| Defect | Cooperate | 5 | 0 |
| Defect | Defect | 1 | 1 |

These payoffs define the incentives used throughout the simulation.

## Opponent Strategies

The simulation compares the adaptive agent against nine hand-written strategies.

| Strategy | Behavior |
|---|---|
| Always Cooperate | Cooperates every round. |
| Always Defect | Defects every round. |
| Random | Selects either action with equal probability. |
| Tit-for-Tat | Starts by cooperating, then copies the opponent's previous action. |
| Suspicious Tit-for-Tat | Starts by defecting, then copies the opponent's previous action. |
| Generous Tit-for-Tat | Usually copies the opponent but has a 20% chance of forgiving a defection. |
| Tit-for-Two-Tats | Defects when the opponent defects in two consecutive rounds. |
| Nine-Move Trigger | Defects while a defection remains in the opponent's stored nine-action history. |
| Action-Comparison Heuristic | Uses comparisons between the players' previous actions to determine its next move. |

These strategies provide different behavioral patterns against which the neural-network agent can be evaluated.

The Nine-Move Trigger is a limited-memory approximation, not a true permanent Grim Trigger strategy.

The Action-Comparison Heuristic is the current implementation labeled Pavlov / Win-Stay-Lose-Shift in the source code. It is not a standard payoff-based WSLS implementation.

## Running the Simulation

### Requirements

- Python 3
- NumPy

Install NumPy:

```bash
python -m pip install numpy
```

### Execute

From the repository root:

```bash
cd "Game Theory"
python Simulation_Runner.py
```

The simulation runs 40,000 rounds against each opponent strategy.

The current configuration uses:

```python
ROUNDS = 40000
LEARNING_RATE = 0.30
SEED = 42
```

### Output

After each matchup, the simulation reports:

- Agent's cumulative reward.
- Opponent's cumulative reward.
- Average reward per round for each player.
- Agent's cooperation rate.
- Agent's defection rate.

The final comparison summarizes these metrics across the opponent strategies.

## Project Structure

```text
GameTheroyBot/
├── Game Theory/
│   ├── LrBot.py
│   ├── Simulation_Runner.py
│   └── Strategies.py
│
├── README.md
└── .gitattributes
```

The repository also contains earlier neural-network prototypes in `Nerual Network Test/`.

These files are developmental experiments and are not imported by the main simulation.

## Limitations and Future Work

The current implementation provides a foundation for studying reward-driven behavior in repeated games, but several limitations remain.

### Current Limitations

- **Deterministic action selection:** The sigmoid output is rounded rather than sampled from a probability distribution.
- **Reward-target learning:** The network is trained toward the previous round's normalized reward rather than directly optimizing a formal policy objective.
- **Limited memory:** The agent only receives the previous nine actions from each player.
- **Single-run evaluation:** The simulation currently uses one configured random seed rather than averaging results over multiple independent trials.
- **Simplified opponent models:** Some hand-written strategies use limited-memory or custom rules rather than their complete classical definitions.
- **No explicit future-reward model:** The network does not calculate the expected long-term value of alternative actions.

### Potential Improvements

1. Investigate stochastic action selection and compare it with the current deterministic approach.
2. Implement a formally defined policy-gradient algorithm as a separate learning experiment.
3. Evaluate the agent over multiple random seeds and report average performance and variability.
4. Track cooperation rates and cumulative rewards throughout individual simulations.
5. Implement persistent memory for strategies that require long-term behavioral tracking.
6. Add automated tests for neural-network updates, action selection, and opponent strategies.
7. Investigate alternative network architectures and input representations.

No performance advantage over the comparison strategies is claimed. Further experimentation is required to determine how the agent behaves under different initializations and evaluation conditions.

## Author

Jeremie-design

[GitHub Repository](https://github.com/Jeremie-design/GameTheroyBot)
