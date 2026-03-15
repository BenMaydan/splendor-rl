import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Tianshou imports
from tianshou.data import Batch
from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.policy import PPOPolicy, MultiAgentPolicyManager
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.utils import TensorboardLogger

# Import your custom environment here
# from splendor_env import SplendorEnv 

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

def train_splendor():
    # 1. Setup Environment Generators
    # Tianshou's PettingZooEnv wrapper natively handles the AEC cycle
    def env_generator():
        env = SplendorEnv(num_players=2, render_mode=None)
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
    critic_net = SplendorActionMaskNet(1, device=device).to(device)
    
    actor = Actor(actor_net, action_shape, device=device).to(device)
    critic = Critic(critic_net, device=device).to(device)

    # 4. Setup Optimizer and PPO Policy
    optim = torch.optim.Adam(set(actor.parameters()).union(critic.parameters()), lr=3e-4)
    
    def dist_fn(logits):
        return torch.distributions.Categorical(logits=logits)
        
    policy = PPOPolicy(
        actor, 
        critic, 
        optim, 
        dist_fn, 
        action_space=sample_env.action_space,
        deterministic_eval=True,
        action_scaling=False
    )

    # 5. Multi-Agent Parameter Sharing
    # We tell Tianshou that all agents in the environment share the exact same policy
    agents = sample_env.agents
    multi_agent_policy = MultiAgentPolicyManager([policy for _ in range(len(agents))], sample_env)

    # 6. Setup Data Collectors
    # Collectors handle the actual playing of the game and storing transitions
    from tianshou.data import Collector, VectorReplayBuffer
    train_collector = Collector(
        multi_agent_policy, 
        train_envs, 
        VectorReplayBuffer(20000, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(multi_agent_policy, test_envs, exploration_noise=False)

    # 7. Setup TensorBoard Logging
    log_path = os.path.join("log", "splendor_ppo")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    # 8. Execute the Training Loop
    print(f"Starting training on {device}...")
    result = onpolicy_trainer(
        multi_agent_policy,
        train_collector,
        test_collector,
        max_epoch=50,
        step_per_epoch=10000,
        repeat_per_collect=10,
        episode_per_test=10,
        batch_size=256,
        step_per_collect=2000,
        logger=logger,
    )

    print(f"Training completed. Final metrics: {result}")
    
    # Save the shared policy
    torch.save(policy.state_dict(), "splendor_shared_policy.pth")
    print("Model saved to splendor_shared_policy.pth")

if __name__ == "__main__":
    train_splendor()
