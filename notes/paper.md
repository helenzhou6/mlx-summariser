# Notes

## Learning to summarize from human feedback (2020 paper)
[Link to paper](https://arxiv.org/pdf/2009.01325)

### Dataset released
The dataset contains 64,832 summary comparisons on the TL;DR dataset, as well as our evaluation data on both TL;DR
(comparisons and Likert scores) and CNN/DM (Likert scores).

### Reward model (RM) = Training a model to optimise for human preferences:
1. Collected a large, high-quality dataset of human comparisons between summaries / a dataset of human preferences between pairs of summaries
2. train a reward model (RM) via supervised learning to predict the human-preferred summary
3. use that model as a reward function to fine-tune a summarization policy using reinforcement learning

### Training a Policy via reinforcement learning (RL)
Finally, we train a policy via reinforcement learning (RL) to maximize the score given by the RM
1. the policy generates a token of text at each ‘time step’, 
2. is updated using the PPO algorithm based on the RM ‘reward’ given to the entire generated summary
3.  We can then gather more human data using samples from the resulting policy, and repeat the process

## High level architecture
0. Start with initial policy that is fine-tuned via supervised learning on the desired dataset.  Then 2 seps repeated iteratively

### Paper side notes
- Side note: "We hope the evidence from our paper motivates machine
learning researchers to pay closer attention to how their training loss affects the
model behavior they actually want."
- SCARY: Summaries from our human feedback models are preferred by our labelers to the original human demonstrations in the dataset
- Previously: fine tuned using supervised learning. Aim: train language models on objectives that more
closely capture the behavior we care about. ROUGE has received criticism for poor correlation with human judgements. RM outperforms other metrics e.g. ROGUE at predicting human preferences


## Acronyms and definitions
- **NLP** = natural language processing
- **Supervised learning**: maching learning where model is trained using labelled data (input-output pairs where correct answer known) - therefore algorithm leans mapping from inputs/features to outputs/labels
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** is a set of metrics used to evaluate the quality of machine-generated text (like summaries) by comparing it to human-written references, based on overlapping units such as n-grams, word sequences, and word pairs.
- **reward learning**: a machine learning technique where an agent learns to make decisions by optimizing a reward signal, which represents how good or desirable an outcome is.  Instead of directly learning from labeled examples, the agent learns what to do by receiving feedback (rewards or penalties) based on its actions, often used in reinforcement learning and human feedback systems.
    - **continuous reward signal** = how good or desirable an outcome is. Output is a scalar value
    - Usually loss function is mean squared error (MSE) or ranking loss if used in preference learning.
    - Supervision type: Can be supervised (e.g., human-labeled reward scores) or used in reinforcement learning to guide policy updates.
    - Reward modeling is about scoring how good something is — even if there's no fixed "right" answer. VS Classification is about deciding what something is.
- **reinforcement learning (RL)**: is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties. The goal is to learn a policy that maximizes cumulative reward over time by choosing the best actions in different situations.
- **policy (π)** = a strategy or function that defines the agent’s behavior — it maps states of the environment to actions the agent should take.
    - Formally: Policy (π) is a function π(a | s) that gives the probability of taking action a when in state s.
    - Two types: **Deterministic policy:** Always chooses the same action for a given state. VS **Stochastic policy:** Assigns probabilities to possible actions for a given state.
    - Goal in RL is to find an optimal policy that maximizes expected cumulative rewards.
- **Cumulative rewards**: In RL, refers to the total amount of reward an agent collects over time as it interacts with the environment.