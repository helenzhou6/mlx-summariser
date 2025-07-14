# Notes

## Learning to summarize from human feedback (2020 paper)
[Link to paper](https://arxiv.org/pdf/2009.01325)

## Datasets
#### Released one
The dataset contains 64,832 summary comparisons on the TL;DR dataset, as well as our evaluation data on both TL;DR
(comparisons and Likert scores) and CNN/DM (Likert scores).
#### Original
1. We use the TL;DR summarization dataset, which contains ~3 million posts from reddit.com across a variety of topics (subreddits), as well summaries of the posts written by the original poster (TL;DRs)
2. sampled & filtered to include only posts where the human-written summaries contain between 24 and 48 tokens
3. Final filtered dataset contains 123,169 posts, and we hold out ~5% as a validation set.

## Overall aim
- define our ground-truth task as producing a model that generates summaries fewer than 48 tokens long that are as good as possible, according to our judgments

## Reinforcement Learning from Human Feedback (RLHF) architecture
<img width="1018" height="694" alt="Screenshot 2025-07-14 at 14 39 10" src="https://github.com/user-attachments/assets/27513f07-6918-4215-96ee-a02d8d99e3f8" />

-  All models are Transformer decoders (in the style of GPT-3) - since support autoregressive text generation
Needed:
- Start with pretrained models to autoregressively predict the next token in a large text corpus. These models are 'zero-shot' baselines (by padding the context with examples of high-quality summaries from the dataset.?? ) 
    - Above technique more like few-shot learning (see below glossary) BUT called **zero shot** since model weights aren't updated, only the input modified.
- Supervised baselines: fine-tune models via supervised learning to predict summarised.
    - Supervised models are used to sample initial summaries for collecting comparisons, initialise policy + erward models and baselines for evaluation 
- The reward model, policy, and value function are the same size.
- Controlling for summary length needs to be taken into consideration (trade-off between conciseness and coverage)

## High level architecture
<img width="1155" height="628" alt="Screenshot 2025-07-14 at 13 03 36" src="https://github.com/user-attachments/assets/7c72ba7a-5394-4c23-a5e9-3efe03d2db0b" />

0. Start with initial policy that is fine-tuned via supervised learning on the desired dataset.  Then 3 steps repeated iteratively:
- Step 1: Collect samples from existing policies and send comparisons to human evaluators. Human evaluator/labeller manually selects best summary out of the two.
- Step 2: Train a reward model to predict the log odds that this summary is better
- Step 3: Optimise a policy against the RM. Use logit output of RM to optimise using reinforcement learning (specifically PPO algorithm) 

### Reward model (RM) = Training a model to optimise for human preferences:
1. Collected a large, high-quality dataset of human comparisons between summaries / a dataset of human preferences between pairs of summaries
    - And started from a supervised baseline.
    - Then added randomly initialised linear head that outputs scalar value
- Also We initialize the value function to the parameters of the reward model.
2. train a reward model (RM) via supervised learning to predict the human-preferred summary
    - i.e. train to predict which summary is better as judged by a human, using specific loss function
    - At end of training, we normalize the reward model outputs such that the reference summaries from our dataset achieve a mean score of 0 (i.e. center the rewards so avoid biased the optimisation towards increasing/decreasing reward values // counteract reward scores being too high or too low on average)
3. use that model as a reward function to fine-tune a summarization policy using reinforcement learning

### Training a Human feedback Policy via reinforcement learning (RL)
Finally, we train a policy via reinforcement learning (RL) to maximize the score given by the RM - therefore model generates higher quality outputs. Policy = deciding actions
0. We initialize our policy to be the model fine-tuned on Reddit TL;DR
1. the policy generates a token of text at each ‘time step’, 
2. is updated using the PPO algorithm based on the RM ‘reward’ given to the entire generated summary
    - Treat the output of the reward model as a reward for the entire summary that we maximize with the PPO algorithm (where each time step is a BPE token)
3.  We can then gather more human data using samples from the resulting policy, and repeat the process
Note: include a term in the reward that penalizes from too much diveragance between RL policy and params (using KL divergence to quantify that deviation)
    - KL terms acts as entropy bonus to ensure policy doesn't learn to produce outputs that are too different from those RM has seen during training. Plus stops from collapsing to single mode

### PPO value function
- Uses a Trasnfromer with completely separate params from the policy 
    - Note PPO has two main components - **policy network**: decides what actions to take, **value function**: estimates how good a given state is // predicts expected future rewards. The policy and value function have different roles and might need to learn different features.
    - Keeping params sepearate allows each network to specialise without interring. 
- PPO loss function that maps the scalar value to the multiple tokens (i.e. tokens/words across the sentence - e.g. "the" shouldn't be punished that much)
- Also called **reward shaping**

### Paper side notes
- Side note: "We hope the evidence from our paper motivates machine
learning researchers to pay closer attention to how their training loss affects the
model behavior they actually want."
- SCARY: Summaries from our human feedback models are preferred by our labelers to the original human demonstrations in the dataset
- Previously: fine tuned using supervised learning. Aim: train language models on objectives that more
closely capture the behavior we care about. ROUGE has received criticism for poor correlation with human judgements. RM outperforms other metrics e.g. ROGUE at predicting human preferences

## Glossary: Acronyms and definitions etc
- **NLP** = natural language processing
- **Supervised learning**: maching learning where model is trained using labelled data (input-output pairs where correct answer known) - therefore algorithm leans mapping from inputs/features to outputs/labels
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** is a set of metrics used to evaluate the quality of machine-generated text (like summaries) by comparing it to human-written references, based on overlapping units such as n-grams, word sequences, and word pairs.
- **reward learning**: a machine learning technique where an agent learns to make decisions by optimizing a reward signal, which represents how good or desirable an outcome is.  Instead of directly learning from labeled examples, the agent learns what to do by receiving feedback (rewards or penalties) based on its actions, often used in reinforcement learning and human feedback systems.
    - **continuous reward signal** = how good or desirable an outcome is. Output is a scalar value
    - Usually loss function is mean squared error (MSE) or ranking loss if used in preference learning.
    - Supervision type: Can be supervised (e.g., human-labeled reward scores) or used in reinforcement learning to guide policy updates.
    - Reward modeling is about scoring how good something is — even if there's no fixed "right" answer. VS Classification is about deciding what something is.
- **reinforcement learning (RL)**: is a type of machine learning where an agent learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties. The goal is to learn a policy that maximizes cumulative reward over time by choosing the best actions in different situations.
    - **Proximal Policy Optimization (PPO)** is a reinforcement learning (RL) algorithm that helps an agent learn how to act in an environment to maximize rewards, but does so in a way that prevents the policy from changing too drastically, which makes training more stable and efficient.
- **policy (π)** = a strategy or function that defines the agent’s behavior — it maps states of the environment to actions the agent should take.
    - Formally: Policy (π) is a function π(a | s) that gives the probability of taking action a when in state s.
    - Two types: **Deterministic policy:** Always chooses the same action for a given state. VS **Stochastic policy:** Assigns probabilities to possible actions for a given state.
    - Goal in RL is to find an optimal policy that maximizes expected cumulative rewards.
- **Cumulative rewards**: In RL, refers to the total amount of reward an agent collects over time as it interacts with the environment.
- **Autoregressive** in text generation = model generates text one token at a time and next one based on all perviously generated tokens
- **zero-shot**: model is not fine tuned on a specific task, expected to perform without additional training
- **baseline**: reference point to evaluate performance of other models - see how well pretrained model performs 
- **few-shot learning** = type of machine learning where a model learns to perform a task using only a small number of examples (called “shots”) provided at inference time, not during training. E.g. What is the capital of France/Germany/Italy?
- **Kullback–Leibler (KL) divergence** = a measure of how one probability distribution differs from another.
- **Entropy bonus** is a term added to the reinforcement learning objective to encourage the agent to maintain exploration by producing more uncertain or diverse actions rather than collapsing to **deterministic behavior** (i.e. model/agent produces same output given same input)
- **supervised fine-tuning (SFT)** = first stage in the RLHF pipeline where a pretrained language model is fine tuned using human labelled examples in a supervised way