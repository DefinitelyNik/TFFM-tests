import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ==================== Configuration ====================
class ModelConfig:
    def __init__(self):
        self.use_attention_gates = True
        self.attention_gate_levels = [0, 1, 2, 3]

        # Multi-level TFFM configuration
        self.use_multi_level_tffm = True
        self.tffm_levels = [0, 1, 2, 3, 4]  # Which decoder levels to apply TFFM

        # TFFM hyperparameters
        # Format: {level: (grid_size, hidden_dim, k_neighbors)}
        self.tffm_configs = {
            0: (20, 24, 5),  # grid: 16→20, hidden: 32→24
            1: (24, 32, 7),  # grid: 20→24, hidden: 48→32
            2: (28, 48, 9),  # grid: 24→28, hidden: 64→48
            3: (32, 48, 12), # grid: 28→32, hidden: 64→48
            4: (32, 64, 15), # grid: 32→32, hidden: 48→64
        }

        # Model settings
        self.encoder_name = "efficientnet_b0"
        self.pretrained = True
        self.decoder_channels = [256, 128, 64, 32, 16]


# ==================== Graph Attention ====================
class GraphAttentionLayer(nn.Module):
    """Enhanced Graph Attention Layer with feature-based reasoning"""

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Learnable temperature for softmax
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, adj_mask=None):
        """
        Args:
            x: [B, N, in_features] node features
            adj_mask: [B, N, N] optional adjacency mask (binary or continuous)
        """
        B, N, _ = x.shape

        # Transform features
        h = self.W(x)  # [B, N, out_features]

        # Self-attention mechanism
        # Algebraically equivalent additive GAT scores. Avoid allocating
        # [B,N,N,2F]; keep parameter names/shapes for checkpoint compatibility.
        # FP rounding may differ from the original concatenated Linear.
        a_left, a_right = self.a.weight.split(self.out_features, dim=1)
        left = F.linear(h, a_left)
        right = F.linear(h, a_right)
        e = self.leakyrelu(left + right.transpose(1, 2))  # [B,N,N]

        # Apply adjacency mask if provided
        if adj_mask is not None:
            e = e * adj_mask

        # Normalized attention
        attention = F.softmax(e / self.temperature, dim=-1)
        attention = self.dropout(attention)

        # Aggregate features
        h_prime = torch.bmm(attention, h)  # [B, N, out_features]

        return F.elu(h_prime)


# ==================== TFFM ====================
class TFFMBlock(nn.Module):
    """
    Topology Aware Feature Fusion Module
    """

    def __init__(self, in_channels, grid_size=24, hidden_dim=64, k_neighbors=9):
        super().__init__()
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors

        # Channel compression
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Multi-scale feature extraction
        self.pool_main = nn.AdaptiveAvgPool2d(grid_size)
        self.pool_aux = nn.AdaptiveAvgPool2d(
            max(grid_size // 2, 4)
        )  # Ensure min size of 4

        # Graph Attention layers with residual connections
        self.gat1 = GraphAttentionLayer(hidden_dim, hidden_dim)
        self.gat2 = GraphAttentionLayer(hidden_dim, hidden_dim)

        # Multi-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Feature enrichment
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Spatial attention for vessel emphasis
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, 1),
            nn.Sigmoid(),
        )

        # Vesselness gating
        self.vessel_gate = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Sigmoid(),
        )

        # Channel expansion
        self.expand = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # Adaptive fusion
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1), nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def _build_feature_graph(self, features, k_neighbors):
        """
        Build graph based on FEATURE SIMILARITY, not spatial coordinates
        """
        B, N, C = features.shape

        # Normalize features for cosine similarity
        features_norm = F.normalize(features, p=2, dim=-1)  # [B, N, C]

        # Compute cosine similarity
        similarity = torch.bmm(
            features_norm, features_norm.transpose(1, 2)
        )  # [B, N, N]

        # Get top-k most similar nodes (feature-based neighbors)
        _, knn_idx = torch.topk(similarity, k_neighbors, dim=-1)  # [B, N, k]

        # Build adjacency matrix
        adj = torch.zeros(B, N, N, device=features.device)
        batch_idx = (
            torch.arange(B, device=features.device)
            .view(B, 1, 1)
            .expand(B, N, k_neighbors)
        )
        node_idx = (
            torch.arange(N, device=features.device)
            .view(1, N, 1)
            .expand(B, N, k_neighbors)
        )

        adj[batch_idx, node_idx, knn_idx] = 1.0

        # Add self-connections
        self_loop = torch.eye(N, device=features.device).unsqueeze(0).expand(B, -1, -1)
        adj = adj + self_loop
        adj = torch.clamp(adj, 0, 1)  # Binary adjacency

        return adj

    def forward(self, x):
        B, C, H, W = x.shape

        # Compress channels
        compressed = self.compress(x)  # [B, hidden_dim, H, W]

        # Multi-scale pooling
        main_features = self.pool_main(compressed)  # [B, hidden_dim, grid, grid]
        aux_features = self.pool_aux(compressed)  # [B, hidden_dim, grid//2, grid//2]

        # Process main scale with graph attention
        main_flat = main_features.view(B, self.hidden_dim, -1).permute(
            0, 2, 1
        )  # [B, N, hidden_dim]

        # Build graph from FEATURES with configurable k_neighbors
        adj_main = self._build_feature_graph(main_flat, self.k_neighbors)

        # Apply graph attention
        graph_out1 = self.gat1(main_flat, adj_main)
        graph_out2 = self.gat2(graph_out1, adj_main)  # [B, N, hidden_dim]

        # Reshape back
        graph_features = graph_out2.permute(0, 2, 1).view(
            B, self.hidden_dim, self.grid_size, self.grid_size
        )

        # Process auxiliary scale
        aux_up = F.interpolate(
            aux_features,
            size=(self.grid_size, self.grid_size),
            mode="bilinear",
            align_corners=False,
        )

        # Fuse multi-scale features
        fused_features = torch.cat([graph_features, aux_up], dim=1)
        fused_features = self.scale_fusion(
            fused_features
        )  # [B, hidden_dim, grid, grid]

        # Apply attention mechanisms
        channel_attn = self.channel_attn(fused_features)
        spatial_attn = self.spatial_attn(fused_features)

        attended_features = fused_features * channel_attn * spatial_attn

        # Apply vesselness gate
        vessel_weights = self.vessel_gate(attended_features)
        attended_features = attended_features * vessel_weights

        # Upsample to original spatial size
        upsampled = F.interpolate(
            attended_features, size=(H, W), mode="bilinear", align_corners=False
        )

        # Expand channels
        expanded = self.expand(upsampled)  # [B, C, H, W]

        # Adaptive fusion with original features
        fusion_weight = self.fusion_gate(torch.cat([x, expanded], dim=1))
        output = self.fusion(torch.cat([x, expanded], dim=1))

        # Residual connection with gating
        output = fusion_weight * output + (1 - fusion_weight) * x

        return output


# ==================== Core ====================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class AttentionGate(nn.Module):
    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2

        self.W_g = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(inter_channels),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm2d(inter_channels),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(
                gate, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        g1 = self.W_g(gate)
        x1 = self.W_x(skip)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return skip * psi


class UNetPPNode(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        total_in = sum(in_channels_list)
        self.conv_block = ConvBlock(total_in, out_channels)

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        x = self.conv_block(x)
        return x


# ==================== Main Architecture ====================
class TFFMSegmentationModel(nn.Module):
    def __init__(self, config=None, num_classes=2):
        super().__init__()
        self.config = config or ModelConfig()
        self.num_classes = num_classes

        # ===== Encoder =====
        self.encoder = timm.create_model(
            self.config.encoder_name,
            pretrained=self.config.pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        enc_channels = self.encoder.feature_info.channels()
        dec_channels = self.config.decoder_channels

        # ===== Bottleneck =====
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[4], dec_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_channels[0]),
            nn.ReLU(inplace=True),
        )

        # ===== Multi-Level TFFM Integration =====
        self.tffm_modules = nn.ModuleDict()
        if self.config.use_multi_level_tffm:
            for level in self.config.tffm_levels:
                grid_size, hidden_dim, k_neighbors = self.config.tffm_configs[level]

                # Determine input channels for each level
                if level == 4:
                    in_channels = dec_channels[0]  # Bottleneck
                else:
                    in_channels = dec_channels[4 - level]  # Decoder channels

                self.tffm_modules[f"level_{level}"] = TFFMBlock(
                    in_channels=in_channels,
                    grid_size=grid_size,
                    hidden_dim=hidden_dim,
                    k_neighbors=k_neighbors,
                )

                print(
                    f"TFFM Level {level}: channels={in_channels}, grid={grid_size}, "
                    f"hidden={hidden_dim}, k={k_neighbors}"
                )

        # ===== Attention Gates =====
        if self.config.use_attention_gates:
            self.attention_gates = nn.ModuleDict(
                {
                    f"level{i}": AttentionGate(dec_channels[4 - i - 1], enc_channels[i])
                    for i in self.config.attention_gate_levels
                }
            )

        # ===== U-Net++ Decoder =====
        self.node_3_0 = ConvBlock(enc_channels[3], dec_channels[1])
        self.node_3_1 = UNetPPNode([enc_channels[3], dec_channels[0]], dec_channels[1])

        self.node_2_0 = ConvBlock(enc_channels[2], dec_channels[2])
        self.node_2_1 = UNetPPNode([enc_channels[2], dec_channels[1]], dec_channels[2])
        self.node_2_2 = UNetPPNode(
            [enc_channels[2], dec_channels[1], dec_channels[2]], dec_channels[2]
        )

        self.node_1_0 = ConvBlock(enc_channels[1], dec_channels[3])
        self.node_1_1 = UNetPPNode([enc_channels[1], dec_channels[2]], dec_channels[3])
        self.node_1_2 = UNetPPNode(
            [enc_channels[1], dec_channels[2], dec_channels[3]], dec_channels[3]
        )
        self.node_1_3 = UNetPPNode(
            [enc_channels[1], dec_channels[2], dec_channels[3], dec_channels[3]],
            dec_channels[3],
        )

        self.node_0_0 = ConvBlock(enc_channels[0], dec_channels[4])
        self.node_0_1 = UNetPPNode([enc_channels[0], dec_channels[3]], dec_channels[4])
        self.node_0_2 = UNetPPNode(
            [enc_channels[0], dec_channels[3], dec_channels[4]], dec_channels[4]
        )
        self.node_0_3 = UNetPPNode(
            [enc_channels[0], dec_channels[3], dec_channels[4], dec_channels[4]],
            dec_channels[4],
        )
        self.node_0_4 = UNetPPNode(
            [
                enc_channels[0],
                dec_channels[3],
                dec_channels[4],
                dec_channels[4],
                dec_channels[4],
            ],
            dec_channels[4],
        )

        # ===== Output Head =====
        self.final_conv = nn.Conv2d(dec_channels[4], num_classes, 1)

    def apply_attention_gate(self, level, gate_features, skip_features):
        if self.config.use_attention_gates and f"level{level}" in self.attention_gates:
            return self.attention_gates[f"level{level}"](gate_features, skip_features)
        return skip_features

    def apply_tffm(self, level, features):
        """Apply TFFM if configured for this level"""
        if self.config.use_multi_level_tffm and f"level_{level}" in self.tffm_modules:
            return self.tffm_modules[f"level_{level}"](features)
        return features

    def forward(self, x):
        input_size = x.shape[2:]
        encoder_features = self.encoder(x)
        enc_0, enc_1, enc_2, enc_3, enc_4 = encoder_features

        # Bottleneck with TFFM (Level 4)
        x4_0 = self.bottleneck(enc_4)
        x4_0 = self.apply_tffm(4, x4_0)  # Apply TFFM at bottleneck

        # Level 3
        x3_0 = self.node_3_0(enc_3)
        enc_3_att = self.apply_attention_gate(3, x4_0, enc_3)
        x3_1 = self.node_3_1(
            enc_3_att,
            F.interpolate(x4_0, scale_factor=2, mode="bilinear", align_corners=False),
        )
        x3_1 = self.apply_tffm(3, x3_1)  # Apply TFFM at level 3

        # Level 2
        x2_0 = self.node_2_0(enc_2)
        enc_2_att = self.apply_attention_gate(2, x3_1, enc_2)
        x2_1 = self.node_2_1(
            enc_2_att,
            F.interpolate(x3_1, scale_factor=2, mode="bilinear", align_corners=False),
        )
        x2_1 = self.apply_tffm(2, x2_1)  # Apply TFFM at level 2

        enc_2_att2 = self.apply_attention_gate(2, x3_1, enc_2)
        x2_2 = self.node_2_2(
            enc_2_att2,
            F.interpolate(x3_1, scale_factor=2, mode="bilinear", align_corners=False),
            x2_1,
        )
        x2_2 = self.apply_tffm(2, x2_2)  # Apply TFFM at level 2

        # Level 1
        x1_0 = self.node_1_0(enc_1)
        enc_1_att = self.apply_attention_gate(1, x2_1, enc_1)
        x1_1 = self.node_1_1(
            enc_1_att,
            F.interpolate(x2_1, scale_factor=2, mode="bilinear", align_corners=False),
        )
        x1_1 = self.apply_tffm(1, x1_1)  # Apply TFFM at level 1

        enc_1_att2 = self.apply_attention_gate(1, x2_2, enc_1)
        x1_2 = self.node_1_2(
            enc_1_att2,
            F.interpolate(x2_2, scale_factor=2, mode="bilinear", align_corners=False),
            x1_1,
        )
        x1_2 = self.apply_tffm(1, x1_2)  # Apply TFFM at level 1

        enc_1_att3 = self.apply_attention_gate(1, x2_2, enc_1)
        x1_3 = self.node_1_3(
            enc_1_att3,
            F.interpolate(x2_2, scale_factor=2, mode="bilinear", align_corners=False),
            x1_1,
            x1_2,
        )
        x1_3 = self.apply_tffm(1, x1_3)  # Apply TFFM at level 1

        # Level 0
        x0_0 = self.node_0_0(enc_0)
        enc_0_att = self.apply_attention_gate(0, x1_1, enc_0)
        x0_1 = self.node_0_1(
            enc_0_att,
            F.interpolate(x1_1, scale_factor=2, mode="bilinear", align_corners=False),
        )
        x0_1 = self.apply_tffm(0, x0_1)  # Apply TFFM at level 0

        enc_0_att2 = self.apply_attention_gate(0, x1_2, enc_0)
        x0_2 = self.node_0_2(
            enc_0_att2,
            F.interpolate(x1_2, scale_factor=2, mode="bilinear", align_corners=False),
            x0_1,
        )
        x0_2 = self.apply_tffm(0, x0_2)  # Apply TFFM at level 0

        enc_0_att3 = self.apply_attention_gate(0, x1_3, enc_0)
        x0_3 = self.node_0_3(
            enc_0_att3,
            F.interpolate(x1_3, scale_factor=2, mode="bilinear", align_corners=False),
            x0_1,
            x0_2,
        )
        x0_3 = self.apply_tffm(0, x0_3)  # Apply TFFM at level 0

        enc_0_att4 = self.apply_attention_gate(0, x1_3, enc_0)
        x0_4 = self.node_0_4(
            enc_0_att4,
            F.interpolate(x1_3, scale_factor=2, mode="bilinear", align_corners=False),
            x0_1,
            x0_2,
            x0_3,
        )
        x0_4 = self.apply_tffm(0, x0_4)  # Apply TFFM at level 0 (final)

        # Final output
        output = self.final_conv(x0_4)
        output = F.interpolate(
            output, size=input_size, mode="bilinear", align_corners=False
        )

        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig()
    config.pretrained = False
    model = TFFMSegmentationModel(config).to(device).eval()
    x = torch.randn(1, 3, 512, 512, device=device)
    with torch.no_grad():
        output = model(x)
    print(f"Input: {x.shape}, Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
