import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        # Ensure groups <= out_channels
        g1 = min(groups, out_channels)
        self.gn1 = nn.GroupNorm(g1, out_channels)
        self.act1 = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        g2 = min(groups, out_channels)
        self.gn2 = nn.GroupNorm(g2, out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x += residual
        return self.act2(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Attention Gate
        F_g: Gate channels (Decoder signal)
        F_l: Label channels (Encoder signal / Skip connection)
        F_int: Intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        # g: Gate (from Decoder)
        # x: Skip (from Encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16, attention=True, depth=3, pool_kernel_size=(2, 2, 2)):
        """
        Configurable 3D U-Net with optional Attention Gates.
        depth: Number of downsampling layers (3 is recommended for 8GB VRAM).
        pool_kernel_size: Tuple (kz, ky, kx). Use (1, 2, 2) for Anisotropic pooling (preserve Z resolution).
        """
        super().__init__()
        self.attention = attention
        self.depth = depth
        
        # --- Encoder ---
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        features = init_features
        curr_in = in_channels
        
        for i in range(depth):
            self.encoders.append(ResidualBlock(curr_in, features))
            self.pools.append(nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_kernel_size))
            curr_in = features
            features *= 2
            
        # --- Bottleneck ---
        self.bottleneck = ResidualBlock(curr_in, features)
        
        # --- Decoder ---
        self.up_convs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        if self.attention:
            self.attentions = nn.ModuleList()
            
        for i in range(depth):
            # Features decrease: 8F -> 4F, etc.
            # UpConv: Input=Features, Output=Features//2
            self.up_convs.append(nn.ConvTranspose3d(features, features // 2, kernel_size=pool_kernel_size, stride=pool_kernel_size))
            
            if self.attention:
                # F_g (Gate) = Features//2 (from UpConv)
                # F_l (Skip) = Features//2 (from Encoder)
                # F_int = Features//4
                self.attentions.append(AttentionBlock(F_g=features//2, F_l=features//2, F_int=features//4))
            
            # Decoder Block: Input = Features (concatenated), Output = Features//2
            self.decoders.append(ResidualBlock(features, features // 2))
            features //= 2

        # --- Final ---
        self.final = nn.Conv3d(init_features, out_channels, kernel_size=1)
        # No activation, return logits

    def forward(self, x):
        # Encoder Pass
        enc_features = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            enc_features.append(x)
            x = self.pools[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder Pass
        for i in range(self.depth):
            # Up
            x = self.up_convs[i](x)
            
            # Skip Connection
            skip = enc_features[self.depth - 1 - i]
            
            # Attention
            if self.attention:
                skip = self.attentions[i](g=x, x=skip)
            
            # Concat
            x = torch.cat([x, skip], dim=1)
            
            # Decode
            x = self.decoders[i](x)
            
        return self.final(x)
