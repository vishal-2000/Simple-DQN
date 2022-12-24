# Mid-Level-Planner
Intent based planner assisting the high level planner
```
Environment: PyBullet
```

```
Use environment.yml to create the required environment
```

## Version 1: Destination Prediction (1-step MDP/Bandit Problem) with DQN
- Observation: RGB-Height map (224*224*4)
- Action space: Pixel-wise Q value (224*224)
### Training method 1
1. Sample 16 action for each state
    1. Out of these 16 actions, 1 action is the pixel with highest Q value
    2. Remaining 15 must be randomly sampled 
    3. Propagate loss for all these 16 pixels, and assign 0 loss to the remaining pixels
### Training method 2
1. Sample only 1 action per state
    1. It's either a random action with some probability
    2. Or the action with highest Q value
## Version 2: Next best action prediction (multi-step MDP) with DQN
- Observation: RGB-Height map (224*224*4)
- Action space: 16 (Q values for pushing by a fixed distance in 16 possible fixed directions)
