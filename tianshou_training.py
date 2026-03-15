import os
import torch
import torch.nn as nn
import numpy as np

# Tianshou imports
from tianshou.data import Batch
from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.algorithm import PPO
from tianshou.algorithm.multiagent.marl import MultiAgentOnPolicyAlgorithm
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.trainer import OnPolicyTrainer, OnPolicyTrainerParams
from tianshou.utils.net.discrete import DiscreteActor, DiscreteCritic
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

# Import your custom environment here
from env.splendor_env import SplendorEnv 

class SplendorFeatureExtractor(nn.Module):
    """
    Flattens the complex dictionary observation space from SplendorEnv into a single 1D tensor,
    ignoring the action mask (which is handled separately by the policy).
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def forward(self, obs, state=None, info={}):
        batch_size = len(obs.phase)
        flat_tensors = []
        
        # Iterate through the dictionary observation Batch
        for key in obs.keys():
            # Tianshou's PettingZooEnv moves 'action_mask' to 'mask'
            if key in ["mask", "action_mask"]:
                continue
            
            val = obs[key]
            
            # Convert NumPy arrays to Tensors
            if isinstance(val, np.ndarray):
                val = torch.tensor(val, dtype=torch.float32, device=self.device)
            elif not isinstance(val, torch.Tensor):
                val = torch.tensor([val], dtype=torch.float32, device=self.device)
            
            # Flatten everything except the batch dimension
            val = val.view(batch_size, -1)
            flat_tensors.append(val)
            
        flat_obs = torch.cat(flat_tensors, dim=1)
        return flat_obs, state


class SplendorActionMaskNet(nn.Module):
    """
    The neural network that processes the flattened state and outputs action logits.
    It automatically applies the action mask by setting invalid actions to -1e9.
    """
    def __init__(self, action_shape, device="cpu"):
        super().__init__()
        self.device = device
        self.extractor = SplendorFeatureExtractor(device)
        
        # LazyLinear automatically calculates the input dimension on the first forward pass.
        self.net = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_shape)
        )

    def forward(self, obs, state=None, info={}):
        # 1. Flatten the dictionary observation
        flat_obs, state = self.extractor(obs, state, info)
        
        # 2. Get raw action logits
        logits = self.net(flat_obs)
        
        # 3. Apply Action Masking
        # Tianshou automatically converts PettingZoo's 'action_mask' into a boolean 'mask'
        if hasattr(obs, "mask"):
            mask = torch.as_tensor(obs.mask, dtype=torch.bool, device=self.device)
            # Replace logits of invalid actions with a massive negative number
            logits[~mask] = -1e9
            
        return logits, state


class SplendorCriticNet(nn.Module):
    """
    The neural network that processes the flattened state specifically for the Critic.
    It outputs a hidden representation to be mapped to V(s) without applying any action masking.
    """
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.extractor = SplendorFeatureExtractor(device)
        
        self.net = nn.Sequential(
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
            # Notice there is no final linear layer here. 
            # Tianshou's DiscreteCritic will automatically add the final Linear(256, 1) mapping.
        )

    def forward(self, obs, state=None, info={}):
        flat_obs, state = self.extractor(obs, state, info)
        hidden = self.net(flat_obs)
        return hidden, state


def train_splendor():
    # 1. Setup Environment Generators
    # Tianshou's PettingZooEnv wrapper natively handles the AEC cycle
    def env_generator():
        env = SplendorEnv(num_players=4, render_mode=None)
        return PettingZooEnv(env)
    
    # We use DummyVectorEnv for parallel processing during collection
    train_envs = DummyVectorEnv([env_generator for _ in range(4)])
    test_envs = DummyVectorEnv([env_generator for _ in range(2)])

    # 2. Setup Device and Dimensions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_env = env_generator()
    action_shape = sample_env.action_space.n

    # 3. Initialize Actor-Critic Networks
    # We use the same base network structure for both, but different instances
    actor_net = SplendorActionMaskNet(action_shape, device=device).to(device)
    critic_net = SplendorCriticNet(device=device).to(device)
    
    actor = DiscreteActor(preprocess_net=actor_net, action_shape=action_shape).to(device)
    critic = DiscreteCritic(preprocess_net=critic_net).to(device)

    # 4. Setup Optimizer and PPO Policy
    optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=3e-4)
    
    def dist_fn(logits):
        return torch.distributions.Categorical(logits=logits)
        
    ppo_algo = PPO(
        actor=actor, 
        critic=critic, 
        optim=optim, 
        dist_fn=dist_fn, 
        action_space=sample_env.action_space,
        deterministic_eval=True,
        action_scaling=False
    )

    # 5. Multi-Agent Parameter Sharing
    # We tell Tianshou that all agents in the environment share the exact same policy
    agents = sample_env.agents
    multi_agent_algo = MultiAgentOnPolicyAlgorithm(
        algorithms=[ppo_algo for _ in range(len(agents))], 
        env=sample_env
    )

    # 6. Setup Data Collectors
    # Collectors handle the actual playing of the game and storing transitions
    train_collector = Collector(
        multi_agent_algo, 
        train_envs, 
        VectorReplayBuffer(20000, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(multi_agent_algo, test_envs, exploration_noise=False)

    # 7. Setup TensorBoard Logging
    log_path = os.path.join("log", "splendor_ppo")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # 8. Execute the Training Loop
    print("Starting training...")
    
    # Instantiate the new params dataclass
    trainer_params = OnPolicyTrainerParams(
        training_collector=train_collector,
        test_collector=test_collector,
        max_epochs=50,
        epoch_num_steps=10000,
        update_step_num_repetitions=10,
        test_step_num_episodes=10,
        batch_size=256,
        collection_step_num_env_steps=2000,
        logger=logger,
    )
    
    # Instantiate the trainer with the algorithm and the params object
    trainer = OnPolicyTrainer(
        algorithm=multi_agent_algo, 
        params=trainer_params
    )
    
    # Execute the training loop
    result = trainer.run()

    print(f"Training completed. Final metrics: {result}")
    
    # Save the shared policy's state dict
    torch.save(ppo_algo.state_dict(), "splendor_shared_policy.pth")
    print("Model saved to splendor_shared_policy.pth")

if __name__ == "__main__":
    train_splendor()
