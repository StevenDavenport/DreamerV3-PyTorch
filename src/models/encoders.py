"""
Text and observation encoders for the skill-conditioned MBRL system.
Includes SBERT integration for goal embedding and observation encoders.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


class SBERTGoalEncoder:
    """SBERT-based goal encoder for embedding natural language skills."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        """
        Initialize SBERT goal encoder.
        
        Args:
            model_name: SBERT model name
            device: Device to use
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def encode_goals(self, goals: List[str]) -> torch.Tensor:
        """
        Encode a list of goal descriptions.
        
        Args:
            goals: List of goal descriptions
            
        Returns:
            goal_embeddings: Goal embeddings [num_goals, embedding_dim]
        """
        embeddings = self.model.encode(goals, convert_to_tensor=True, device=self.device)
        return embeddings
    
    def encode_single_goal(self, goal: str) -> torch.Tensor:
        """
        Encode a single goal description.
        
        Args:
            goal: Goal description
            
        Returns:
            goal_embedding: Goal embedding [embedding_dim]
        """
        embedding = self.model.encode([goal], convert_to_tensor=True, device=self.device)
        return embedding.squeeze(0)
    
    def compute_similarity(self, goal1: str, goal2: str) -> float:
        """
        Compute cosine similarity between two goals.
        
        Args:
            goal1: First goal description
            goal2: Second goal description
            
        Returns:
            similarity: Cosine similarity score
        """
        embedding1 = self.encode_single_goal(goal1)
        embedding2 = self.encode_single_goal(goal2)
        
        similarity = F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        return similarity.item()


class CLIPTextEncoder:
    """CLIP text encoder for goal embedding."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        """
        Initialize CLIP text encoder.
        
        Args:
            model_name: CLIP model name
            device: Device to use
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.embedding_dim = self.model.config.projection_dim
        
    def encode_goals(self, goals: List[str]) -> torch.Tensor:
        """
        Encode a list of goal descriptions using CLIP.
        
        Args:
            goals: List of goal descriptions
            
        Returns:
            goal_embeddings: Goal embeddings [num_goals, embedding_dim]
        """
        # Tokenize
        inputs = self.tokenizer(goals, padding=True, truncation=True, 
                               max_length=77, return_tensors="pt").to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.text_embeds
        
        return embeddings
    
    def encode_single_goal(self, goal: str) -> torch.Tensor:
        """
        Encode a single goal description.
        
        Args:
            goal: Goal description
            
        Returns:
            goal_embedding: Goal embedding [embedding_dim]
        """
        embeddings = self.encode_goals([goal])
        return embeddings.squeeze(0)


class ObservationEncoder(nn.Module):
    """Neural network encoder for observations."""
    
    def __init__(self, obs_dim: int, hidden_dim: int, latent_dim: int, 
                 encoder_type: str = "mlp"):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        
        if encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
        elif encoder_type == "cnn":
            # For image observations
            self.encoder = nn.Sequential(
                nn.Conv2d(obs_dim, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, latent_dim)
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode observations.
        
        Args:
            obs: Observations [batch_size, obs_dim] or [batch_size, channels, height, width]
            
        Returns:
            encoded_obs: Encoded observations [batch_size, latent_dim]
        """
        return self.encoder(obs)


class TransitionDescriptionEncoder:
    """Encoder for transition descriptions (text descriptions of state changes)."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        """
        Initialize transition description encoder.
        
        Args:
            model_name: SBERT model name
            device: Device to use
        """
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
    def encode_transitions(self, descriptions: List[str]) -> torch.Tensor:
        """
        Encode transition descriptions.
        
        Args:
            descriptions: List of transition descriptions
            
        Returns:
            transition_embeddings: Transition embeddings [num_transitions, embedding_dim]
        """
        embeddings = self.model.encode(descriptions, convert_to_tensor=True, device=self.device)
        return embeddings
    
    def compute_goal_transition_similarity(self, goal: str, transition_desc: str) -> float:
        """
        Compute similarity between goal and transition description.
        
        Args:
            goal: Goal description
            transition_desc: Transition description
            
        Returns:
            similarity: Cosine similarity score
        """
        goal_embedding = self.model.encode([goal], convert_to_tensor=True, device=self.device)
        transition_embedding = self.model.encode([transition_desc], convert_to_tensor=True, device=self.device)
        
        similarity = F.cosine_similarity(goal_embedding, transition_embedding)
        return similarity.item()


class MultiModalEncoder(nn.Module):
    """Multi-modal encoder that combines different input types."""
    
    def __init__(self, obs_encoder: ObservationEncoder, 
                 text_encoder: Union[SBERTGoalEncoder, CLIPTextEncoder],
                 fusion_dim: int = 512):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.text_encoder = text_encoder
        self.fusion_dim = fusion_dim
        
        # Fusion network
        total_dim = obs_encoder.latent_dim + text_encoder.embedding_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
    def forward(self, obs: torch.Tensor, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode observation and text jointly.
        
        Args:
            obs: Observations
            text: Text input (goal or description)
            
        Returns:
            joint_encoding: Joint encoding [batch_size, fusion_dim]
        """
        # Encode observation
        obs_encoded = self.obs_encoder(obs)
        
        # Encode text
        if isinstance(text, str):
            text_encoded = self.text_encoder.encode_single_goal(text).unsqueeze(0)
        else:
            text_encoded = self.text_encoder.encode_goals(text)
        
        # Ensure batch dimension matches
        if obs_encoded.shape[0] != text_encoded.shape[0]:
            text_encoded = text_encoded.expand(obs_encoded.shape[0], -1)
        
        # Concatenate and fuse
        combined = torch.cat([obs_encoded, text_encoded], dim=-1)
        joint_encoding = self.fusion_net(combined)
        
        return joint_encoding


def create_goal_encoder(encoder_type: str = "sbert", **kwargs) -> Union[SBERTGoalEncoder, CLIPTextEncoder]:
    """
    Factory function to create goal encoders.
    
    Args:
        encoder_type: Type of encoder ("sbert" or "clip")
        **kwargs: Additional arguments
        
    Returns:
        Goal encoder instance
    """
    if encoder_type == "sbert":
        return SBERTGoalEncoder(**kwargs)
    elif encoder_type == "clip":
        return CLIPTextEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


def compute_cosine_similarity_reward(goal_embedding: torch.Tensor, 
                                   transition_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity reward between goal and transition embeddings.
    
    Args:
        goal_embedding: Goal embedding [batch_size, embedding_dim]
        transition_embeddings: Transition embeddings [batch_size, seq_len, embedding_dim]
        
    Returns:
        rewards: Cosine similarity rewards [batch_size, seq_len]
    """
    # Normalize embeddings
    goal_norm = F.normalize(goal_embedding, p=2, dim=-1)
    transition_norm = F.normalize(transition_embeddings, p=2, dim=-1)
    
    # Compute cosine similarity
    # goal_norm: [batch_size, embedding_dim]
    # transition_norm: [batch_size, seq_len, embedding_dim]
    # We want to compute similarity for each timestep
    goal_expanded = goal_norm.unsqueeze(1)  # [batch_size, 1, embedding_dim]
    
    similarity = torch.sum(goal_expanded * transition_norm, dim=-1)  # [batch_size, seq_len]
    
    return similarity


def generate_transition_descriptions(latent_states: torch.Tensor, 
                                   actions: torch.Tensor,
                                   world_model: nn.Module) -> List[str]:
    """
    Generate text descriptions of transitions.
    This is a placeholder - in practice, you would use a learned model or rule-based system.
    
    Args:
        latent_states: Latent states [batch_size, seq_len, latent_dim]
        actions: Actions taken [batch_size, seq_len, action_dim]
        world_model: World model for decoding observations
        
    Returns:
        descriptions: List of transition descriptions
    """
    # This is a placeholder implementation
    # In practice, you would:
    # 1. Decode latent states to observations
    # 2. Use a learned model or rule-based system to generate descriptions
    # 3. Return meaningful text descriptions
    
    batch_size, seq_len, _ = latent_states.shape
    descriptions = []
    
    for b in range(batch_size):
        for t in range(seq_len):
            # Placeholder description
            desc = f"agent performed action {actions[b, t].argmax().item()} at step {t}"
            descriptions.append(desc)
    
    return descriptions 