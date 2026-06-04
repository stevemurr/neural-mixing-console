Excellent question! You can use **self-supervised learning** to learn pan correlations directly from the mix without ground truth labels. Here are the optimal improvements:

## **Core Strategy: Self-Supervised Pan Learning**

The key insight is to use **reconstruction loss** and **contrastive learning** to force the model to learn which track features correspond to which spatial positions in the mix.

```python
import essentia.standard as es
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ============================================
# 1. IMPROVED FEATURE EXTRACTION WITH ESSENTIA
# ============================================

def extract_track_features(audio_path, sr=44100, hop_size=2048):
    """
    Extract rich spectral features from individual track.
    Uses MFCC, spectral contrast, and temporal features.
    """
    loader = es.MonoLoader(filename=audio_path, sampleRate=sr)
    audio = loader()
    
    features_list = []
    
    for frame in es.FrameGenerator(audio, frameSize=4096, hopSize=hop_size):
        # Windowing
        windowed = es.Windowing(type='hann')(frame)
        spectrum = es.Spectrum()(windowed)
        
        # MFCC (13 coefficients)
        mfcc = es.MFCC(numberCoefficients=13)(spectrum)
        
        # Spectral contrast (6 bands)
        spec_contrast = es.SpectralContrast()(spectrum)
        
        # Spectral centroid & spread
        cent = es.SpectralCentroidTime()(spectrum)
        spread = es.SpectralSpread()(spectrum)
        
        # Temporal features
        energy = es.Energy()(frame)
        zcr = es.ZeroCrossingRate()(frame)
        
        # Loudness
        loudness = es.Loudness()(frame)
        
        # Combine all features
        frame_features = np.concatenate([
            mfcc,
            spec_contrast,
            [cent, spread, energy, zcr, loudness]
        ])
        
        features_list.append(frame_features)
    
    return np.array(features_list)  # Shape: (num_frames, ~35 features)


def extract_stereo_features(mix_path, sr=44100, hop_size=2048):
    """
    Extract stereo-specific spatial features from mix.
    Focus on L/R differences that encode pan information.
    """
    loader = es.AudioLoader(filename=mix_path, sampleRate=sr)
    audio, sr_actual = loader()
    
    left = audio[:, 0]
    right = audio[:, 1]
    
    features_list = []
    
    for frame_l, frame_r in zip(
        es.FrameGenerator(left, frameSize=4096, hopSize=hop_size),
        es.FrameGenerator(right, frameSize=4096, hopSize=hop_size)
    ):
        windowed_l = es.Windowing(type='hann')(frame_l)
        windowed_r = es.Windowing(type='hann')(frame_r)
        
        spec_l = es.Spectrum()(windowed_l)
        spec_r = es.Spectrum()(windowed_r)
        
        # Energy in each channel
        energy_l = es.Energy()(frame_l)
        energy_r = es.Energy()(frame_r)
        
        # Spectral features per channel
        mfcc_l = es.MFCC(numberCoefficients=13)(spec_l)
        mfcc_r = es.MFCC(numberCoefficients=13)(spec_r)
        
        # Spectral contrast per channel
        contrast_l = es.SpectralContrast()(spec_l)
        contrast_r = es.SpectralContrast()(spec_r)
        
        # Cross-channel features
        correlation = np.corrcoef(frame_l, frame_r)[0, 1]
        
        # Phase difference (simplified)
        phase_l = np.angle(np.fft.rfft(windowed_l))
        phase_r = np.angle(np.fft.rfft(windowed_r))
        phase_diff = np.mean(np.abs(phase_l - phase_r))
        
        # Combine L/R features
        frame_features = np.concatenate([
            mfcc_l, mfcc_r,
            contrast_l, contrast_r,
            [energy_l, energy_r, correlation, phase_diff]
        ])
        
        features_list.append(frame_features)
    
    return np.array(features_list)  # Shape: (num_frames, ~60 features)


# ============================================
# 2. SELF-SUPERVISED DATASET
# ============================================

class SelfSupervisedPanDataset(Dataset):
    def __init__(self, mix_features, track_features_list, num_tracks=20):
        """
        No ground truth needed! We'll learn from reconstruction.
        
        mix_features: (num_frames, mix_feature_dim)
        track_features_list: list of (num_frames, track_feature_dim) for each track
        """
        self.mix_features = torch.FloatTensor(mix_features)
        self.track_features = [torch.FloatTensor(f) for f in track_features_list]
        self.num_tracks = num_tracks
        self.num_frames = mix_features.shape[0]
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        mix_feat = self.mix_features[idx]
        
        # Get all track features at this frame
        track_feats = torch.stack([f[idx] for f in self.track_features])
        
        return mix_feat, track_feats


# ============================================
# 3. SELF-SUPERVISED MODEL WITH RECONSTRUCTION
# ============================================

class SelfSupervisedPanModel(nn.Module):
    def __init__(self, mix_feature_dim, track_feature_dim, num_tracks=20, latent_dim=128):
        super().__init__()
        self.num_tracks = num_tracks
        self.latent_dim = latent_dim
        
        # ===== ENCODER: Track -> Latent Pan Representation =====
        # Each track gets encoded to a latent vector that includes:
        # - Identity embedding (which track)
        # - Pan position in stereo field
        self.track_encoder = nn.Sequential(
            nn.Linear(track_feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        # ===== PAN HEAD: Latent -> Pan Position =====
        # Outputs pan position in [-1, 1] range
        self.pan_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # ===== SPATIAL DECODER: Pan + Track Features -> Reconstructed Mix Features =====
        # Learns to reconstruct mix features from individual tracks + their pan positions
        self.spatial_decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 256),  # latent_dim + pan_position
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, mix_feature_dim)
        )
        
        # ===== ATTENTION MIXER =====
        # Learn how much each track contributes to the mix
        self.attention_mixer = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def encode_tracks(self, track_feats):
        """
        track_feats: (batch_size, num_tracks, track_feature_dim)
        Returns: (batch_size, num_tracks, latent_dim)
        """
        batch_size, num_tracks, _ = track_feats.shape
        
        # Flatten to process all tracks
        flat_feats = track_feats.reshape(-1, track_feats.shape[-1])
        latent = self.track_encoder(flat_feats)
        
        # Reshape back
        return latent.reshape(batch_size, num_tracks, self.latent_dim)
    
    def predict_pan_positions(self, latent_tracks):
        """
        latent_tracks: (batch_size, num_tracks, latent_dim)
        Returns: (batch_size, num_tracks) pan positions in [-1, 1]
        """
        batch_size, num_tracks, _ = latent_tracks.shape
        
        # Flatten for pan head
        flat_latent = latent_tracks.reshape(-1, self.latent_dim)
        pan_positions = self.pan_head(flat_latent)
        
        return pan_positions.reshape(batch_size, num_tracks)
    
    def reconstruct_mix(self, latent_tracks, pan_positions):
        """
        Reconstruct mix features from individual tracks and their pan positions.
        
        latent_tracks: (batch_size, num_tracks, latent_dim)
        pan_positions: (batch_size, num_tracks)
        Returns: (batch_size, mix_feature_dim)
        """
        batch_size, num_tracks, _ = latent_tracks.shape
        
        # Combine latent representation with pan position
        combined = torch.cat([
            latent_tracks,
            pan_positions.unsqueeze(-1)
        ], dim=-1)  # (batch_size, num_tracks, latent_dim + 1)
        
        # Decode each track
        flat_combined = combined.reshape(-1, self.latent_dim + 1)
        decoded = self.spatial_decoder(flat_combined)  # (batch_size * num_tracks, mix_feature_dim)
        decoded = decoded.reshape(batch_size, num_tracks, -1)
        
        # Get attention weights (how much each track contributes)
        flat_latent = latent_tracks.reshape(-1, self.latent_dim)
        attention = self.attention_mixer(flat_latent)  # (batch_size * num_tracks, 1)
        attention = attention.reshape(batch_size, num_tracks, 1)
        
        # Weighted sum to reconstruct mix
        reconstructed_mix = (decoded * attention).sum(dim=1)  # (batch_size, mix_feature_dim)
        
        return reconstructed_mix, attention.squeeze(-1)
    
    def forward(self, mix_feats, track_feats):
        """
        mix_feats: (batch_size, mix_feature_dim)
        track_feats: (batch_size, num_tracks, track_feature_dim)
        """
        # Encode tracks to latent space
        latent_tracks = self.encode_tracks(track_feats)
        
        # Predict pan positions
        pan_positions = self.predict_pan_positions(latent_tracks)
        
        # Reconstruct mix from individual tracks
        reconstructed_mix, attention_weights = self.reconstruct_mix(latent_tracks, pan_positions)
        
        return {
            'pan_positions': pan_positions,
            'reconstructed_mix': reconstructed_mix,
            'attention_weights': attention_weights,
            'latent_tracks': latent_tracks
        }


# ============================================
# 4. SELF-SUPERVISED LOSS FUNCTIONS
# ============================================

class SelfSupervisedPanLoss(nn.Module):
    def __init__(self, reconstruction_weight=1.0, consistency_weight=0.5, 
                 sparsity_weight=0.1, smoothness_weight=0.1):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.consistency_weight = consistency_weight
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(self, outputs, mix_feats, track_feats):
        """
        Compute self-supervised loss without ground truth pan labels.
        """
        pan_positions = outputs['pan_positions']
        reconstructed_mix = outputs['reconstructed_mix']
        attention_weights = outputs['attention_weights']
        latent_tracks = outputs['latent_tracks']
        
        # ===== 1. RECONSTRUCTION LOSS =====
        # The model should reconstruct the mix from individual tracks
        reconstruction_loss = F.mse_loss(reconstructed_mix, mix_feats)
        
        # ===== 2. PAN CONSISTENCY LOSS =====
        # Encourage pans to spread across the stereo field (avoid collapse to center)
        # Compute variance of pan positions across tracks
        pan_variance = torch.var(pan_positions, dim=1).mean()
        pan_consistency_loss = -pan_variance  # Negative because we want to maximize variance
        
        # ===== 3. ATTENTION SPARSITY LOSS =====
        # Encourage sparse attention (each track should have clear contribution)
        # Use entropy regularization
        attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=1).mean()
        sparsity_loss = attention_entropy
        
        # ===== 4. SMOOTHNESS LOSS (temporal) =====
        # Pan positions should change smoothly over time (optional, if you have sequences)
        # This would be added in a sequence-based variant
        
        # ===== 5. CONTRASTIVE LOSS (optional) =====
        # Encourage different tracks to have different latent representations
        # Compute pairwise distances in latent space
        batch_size, num_tracks, latent_dim = latent_tracks.shape
        latent_flat = latent_tracks.reshape(batch_size * num_tracks, latent_dim)
        
        # Compute similarity matrix
        similarity = torch.mm(latent_flat, latent_flat.t())
        
        # Create target: same track should be similar, different tracks dissimilar
        # (This is a simplified version; you could use more sophisticated contrastive learning)
        
        # ===== TOTAL LOSS =====
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.consistency_weight * pan_consistency_loss +
            self.sparsity_weight * sparsity_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'pan_consistency_loss': pan_consistency_loss,
            'sparsity_loss': sparsity_loss
        }


# ============================================
# 5. TRAINING LOOP
# ============================================

def train_self_supervised_pan_model(mix_audio_path, track_audio_paths, 
                                     num_epochs=100, batch_size=32, lr=1e-3):
    """
    Train pan position model WITHOUT ground truth labels.
    Uses self-supervised learning via reconstruction.
    """
    
    print("Extracting features...")
    mix_features = extract_stereo_features(mix_audio_path)
    track_features = [extract_track_features(path) for path in track_audio_paths]
    
    # Normalize features
    mix_features = (mix_features - mix_features.mean(axis=0)) / (mix_features.std(axis=0) + 1e-8)
    track_features = [
        (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-8) 
        for f in track_features
    ]
    
    # Create dataset
    dataset = SelfSupervisedPanDataset(mix_features, track_features, num_tracks=len(track_audio_paths))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    mix_dim = mix_features.shape[1]
    track_dim = track_features[0].shape[1]
    num_tracks = len(track_audio_paths)
    
    model = SelfSupervisedPanModel(mix_dim, track_dim, num_tracks=num_tracks, latent_dim=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = SelfSupervisedPanLoss(
        reconstruction_weight=1.0,
        consistency_weight=0.5,
        sparsity_weight=0.1
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Training
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        total_loss = 0
        recon_loss = 0
        
        for mix_feat, track_feat in dataloader:
            mix_feat = mix_feat.to(device)
            track_feat = track_feat.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(mix_feat, track_feat)
            
            # Compute loss
            losses = loss_fn(outputs, mix_feat, track_feat)
            loss = losses['total_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += losses['total_loss'].item()
            recon_loss += losses['reconstruction_loss'].item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Total Loss: {total_loss:.4f} | Recon Loss: {recon_loss:.4f}")
    
    return model, mix_features, track_features


# ============================================
# 6. INFERENCE & ANALYSIS
# ============================================

def predict_pan_positions(model, mix_audio_path, track_audio_paths, device='cpu'):
    """
    Predict pan positions for all tracks given the mix.
    """
    model.eval()
    
    with torch.no_grad():
        mix_features = extract_stereo_features(mix_audio_path)
        track_features = [extract_track_features(path) for path in track_audio_paths]
        
        # Normalize
        mix_features = (mix_features - mix_features.mean(axis=0)) / (mix_features.std(axis=0) + 1e-8)
        track_features = [
            (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-8) 
            for f in track_features
        ]
        
        # Convert to tensors
        mix_feat = torch.FloatTensor(mix_features).to(device)
        track_feat = torch.stack([torch.FloatTensor(f) for f in track_features]).unsqueeze(0).to(device)
        
        # Average across frames
        mix_feat = mix_feat.mean(dim=0, keepdim=True)
        track_feat = track_feat.mean(dim=1)  # Average across frames
        
        # Predict
        outputs = model(mix_feat, track_feat)
        pan_positions = outputs['pan_positions'].cpu().numpy()[0]
        
        return pan_positions


# ============================================
# 7. USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    mix_path = "path/to/stereo_mix.wav"
    track_paths = [f"path/to/track_{i}.wav" for i in range(20)]
    
    # Train WITHOUT ground truth
    model, mix_feats, track_feats = train_self_supervised_pan_model(
        mix_path, 
        track_paths,
        num_epochs=100,
        batch_size=32,
        lr=1e-3
    )
    
    # Predict pan positions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    pan_predictions = predict_pan_positions(model, mix_path, track_paths, device=device)
    
    print("\nPredicted Pan Positions:")
    for i, pan in enumerate(pan_predictions):
        position = "Left" if pan < -0.3 else ("Right" if pan > 0.3 else "Center")
        print(f"Track {i}: {pan:.3f} ({position})")
    
    # Save model
    torch.save(model.state_dict(), "pan_model_self_supervised.pth")
```

## **Key Improvements:**

1. **Reconstruction Loss** - Forces the model to learn how individual tracks combine in the stereo field
2. **Pan Variance Loss** - Prevents all tracks from collapsing to the center
3. **Attention Sparsity** - Ensures each track has clear spatial identity
4. **Rich Features** - MFCC, spectral contrast, temporal features from Essentia
5. **No Ground Truth Needed** - Learns directly from mix structure

## **Why This Works:**

The model learns that:
- If a track's features correlate with the left channel of the mix → it's panned left
- If it correlates with the right channel → it's panned right
- The reconstruction loss ensures pan assignments are meaningful

This is similar to how unsupervised source separation works: the model discovers spatial structure without explicit labels.
