"""GNN-ViT Network: Combines SchNet graph neural network with Vision Transformer."""

import timm
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SchNet, global_add_pool


class GNNViTNetwork(nn.Module):
    """Hybrid network combining graph and image modalities for molecular property prediction."""

    def __init__(
        self,
        schnet_hidden_channels: int = 128,
        schnet_num_filters: int = 128,
        schnet_num_interactions: int = 3,
        schnet_num_gaussians: int = 50,
        schnet_cutoff: float = 10.0,
        vit_img_size: int = 32,
        vit_patch_size: int = 4,
        vit_embed_dim: int = 128,
        vit_depth: int = 4,
        vit_num_heads: int = 4,
        vit_drop_rate: float = 0.1,
        vit_attn_drop_rate: float = 0.1,
        fusion_hidden_dim: int = 64,
        output_dim: int = 1,
    ):
        """
        Args:
            schnet_hidden_channels: Hidden channels in SchNet
            schnet_num_filters: Number of filters in SchNet convolutions
            schnet_num_interactions: Number of interaction blocks in SchNet
            schnet_num_gaussians: Number of Gaussian basis functions
            schnet_cutoff: Cutoff distance for interactions
            vit_img_size: Input image size for ViT
            vit_patch_size: Patch size for ViT
            vit_embed_dim: Embedding dimension for ViT
            vit_depth: Number of transformer blocks
            vit_num_heads: Number of attention heads
            vit_drop_rate: Dropout rate for ViT (MLP and embedding)
            vit_attn_drop_rate: Dropout rate for ViT attention weights
            fusion_hidden_dim: Hidden dimension for fusion MLP
            output_dim: Output dimension (1 for regression)
        """
        super().__init__()

        # Graph encoder: SchNet
        self.schnet = SchNet(
            hidden_channels=schnet_hidden_channels,
            num_filters=schnet_num_filters,
            num_interactions=schnet_num_interactions,
            num_gaussians=schnet_num_gaussians,
            cutoff=schnet_cutoff,
        )

        # Store features from last SchNet interaction
        self.schnet_features = {}

        def get_features(name):
            def hook(model, input, output):
                self.schnet_features[name] = output

            return hook

        # Register hook on the last interaction block
        self.schnet.interactions[-1].register_forward_hook(get_features("last_interaction"))

        # Image encoder: Vision Transformer
        self.vit = timm.models.vision_transformer.VisionTransformer(
            img_size=vit_img_size,
            patch_size=vit_patch_size,
            in_chans=1,
        num_classes=0,  # No classification head, just embeddings
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
        )

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(schnet_hidden_channels + vit_embed_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, output_dim),
        )

    def forward(self, batch: Data):
        """Forward pass combining graph and image data.

        Args:
            batch: PyTorch Geometric batch with .x, .pos, .edge_index, .batch,
                   and .persistence_img [batch_size, 1, H, W]

        Returns:
            Predictions [batch_size] or [batch_size, output_dim]
        """
        # Extract atomic numbers from one-hot encoded features
        atom_indices = batch.x.argmax(dim=1)
        z_values = torch.tensor(
            [1, 6, 7, 8, 16], device=batch.x.device, dtype=torch.long
        )[atom_indices]

        # Process graph through SchNet
        _ = self.schnet(z_values, batch.pos, batch.batch)
        h = self.schnet_features["last_interaction"]

        # Pool node features to graph-level
        graph_embedding = global_add_pool(h, batch.batch)  # [batch_size, hidden_channels]

        # Process persistence images through ViT
        batch_images = batch.persistence_img  # [batch_size, 1, H, W]
        if batch_images.dim() == 3:
            batch_images = batch_images.unsqueeze(1)

        image_embedding = self.vit(batch_images)  # [batch_size, embed_dim]

        # Fusion
        combined = torch.cat([graph_embedding, image_embedding], dim=1)
        prediction = self.fusion_mlp(combined)

        return prediction.squeeze(-1)
