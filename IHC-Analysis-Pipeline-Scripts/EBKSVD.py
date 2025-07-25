#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EBKSVD Stain Vector Estimation Script for IHC Pipeline
------------------------------------------------------
This script implements the Empirical Bayesian K-SVD (EBKSVD) algorithm for stain vector estimation
in histopathology images. It estimates the color deconvolution matrix for the three main stains:
Nuclei (H&E), Myelin (Blue), and Microglia (Brown).

Part of the IHC Pipeline GUI application.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import logging
import multiprocessing
import glob
from scipy.linalg import pinv
from sklearn.decomposition import NMF
import time
import warnings
from functools import partial
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger("EBKSVD")


class ImprovedThreeStainBKSVD:
    """
    Improved Bayesian K-SVD for three-stain separation with robust initialization.
    """
    
    def __init__(self, n_stains=3, sparsity_weight=0.1, beta_prior=10.0, reference_vectors=None):
        """
        Initialize the algorithm with 3-stain specific adjustments.
        
        Args:
            n_stains: Number of stains (default: 3)
            sparsity_weight: Weight for sparsity regularization
            beta_prior: Prior for noise precision
            reference_vectors: Reference stain vectors (3x3 matrix)
        """
        self.n_stains = n_stains
        
        # Adjust sparsity for 3-stain case
        if n_stains == 3:
            sparsity_weight = sparsity_weight * 0.5  # Reduce by half for 3 stains
            
        self.sparsity_weight = sparsity_weight
        self.beta_prior = min(beta_prior, 20.0)  # Cap beta prior
        self.EPS = 1e-10
        
        # Use provided reference vectors or default ones
        if reference_vectors is not None:
            self.reference_vectors = reference_vectors
            print("Using provided reference stain vectors")
            sys.stdout.flush()
        else:
            # Default reference vectors
            self.reference_vectors = np.array([
                [0.9436, 0.2920, 0.1562],  # H&E nuclei
                [0.7306, 0.6608, 0.1722],  # Myelin  
                [0.2161, 0.6447, 0.7332]   # Microglia
            ]).T
            print("Using default reference stain vectors")
            sys.stdout.flush()
        
        # Compute expected dominant channels from reference vectors
        self.expected_dominant = [np.argmax(self.reference_vectors[:, i]) 
                                  for i in range(self.n_stains)]

    def validate_stain_vectors(self, M):
        """Validate that stain vectors make physical sense."""
        issues = []
        
        # Check for negative values (non-physical)
        if np.any(M < -0.1):
            negative_indices = np.where(M < -0.1)
            for i, j in zip(negative_indices[0], negative_indices[1]):
                issues.append(f"Negative value {M[i,j]:.3f} in channel {i}, stain {j}")
        
        # Check for proper color characteristics
        stain_names = ["Nuclei_HE", "Myelin_Blue", "Microglia_Brown"]
        
        for i, (name, expected) in enumerate(zip(stain_names, self.expected_dominant)):
            dominant = np.argmax(M[:, i])
            if dominant != expected:
                issues.append(f"{name} dominant channel is {dominant}, expected {expected}")
        
        # Check normalization
        for i in range(self.n_stains):
            norm = np.linalg.norm(M[:, i])
            if abs(norm - 1.0) > 0.01:
                issues.append(f"Stain {i} not normalized: {norm:.3f}")
        
        # Check for zero vectors
        for i in range(self.n_stains):
            if np.allclose(M[:, i], 0):
                issues.append(f"Stain {i} is zero vector")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _ensure_valid_stain_matrix(self, M):
        """Ensure stain matrix has no zero vectors and is properly normalized."""
        M_fixed = M.copy()
        
        for i in range(self.n_stains):
            # Check if vector is near zero
            if np.linalg.norm(M_fixed[:, i]) < 0.1:
                # Replace with corresponding reference vector
                M_fixed[:, i] = self.reference_vectors[:, i]
                logger.warning(f"Replaced zero stain vector {i} with reference")
            
            # Ensure non-negative
            M_fixed[:, i] = np.maximum(0, M_fixed[:, i])
            
            # Normalize
            norm = np.linalg.norm(M_fixed[:, i])
            if norm > self.EPS:
                M_fixed[:, i] /= norm
            else:
                M_fixed[:, i] = self.reference_vectors[:, i]
        
        return M_fixed
    
    def smart_initialization(self, Y, method='hybrid', n_init=10):
        """Robust initialization with safeguards against degenerate solutions."""
        init_info = {'method': method}
        
        if method == 'reference':
            M_init = self.reference_vectors.copy()
            
        elif method == 'nmf':
            # Robust NMF with multiple attempts
            best_M = None
            best_error = float('inf')
            best_is_valid = False
            
            Y_positive = Y - np.min(Y) + 0.01
            
            for run in range(n_init):
                try:
                    init_method = 'nndsvd' if run == 0 else 'random'
                    
                    nmf = NMF(n_components=self.n_stains, 
                             init=init_method,
                             max_iter=1000,
                             random_state=run, 
                             alpha_W=0.01,
                             alpha_H=0.01,   
                             l1_ratio=0.5,
                             beta_loss='frobenius',
                             solver='mu')
                    
                    C_nmf = nmf.fit_transform(Y_positive.T).T
                    M_nmf = nmf.components_.T
                    
                    M_nmf = self._ensure_valid_stain_matrix(M_nmf)
                    is_valid, issues = self.validate_stain_vectors(M_nmf)
                    
                    reconstruction = M_nmf @ C_nmf
                    error = np.linalg.norm(Y_positive - reconstruction, 'fro')
                    
                    if (is_valid and not best_is_valid) or \
                       (is_valid == best_is_valid and error < best_error):
                        best_error = error
                        best_M = M_nmf
                        best_is_valid = is_valid
                        
                except Exception as e:
                    logger.warning(f"NMF run {run} failed: {e}")
                    continue
            
            if best_M is None:
                logger.warning("All NMF runs failed, using reference vectors")
                M_init = self.reference_vectors.copy()
            else:
                M_init = best_M
                
            init_info['nmf_error'] = best_error
            
        else:  # hybrid
            M_init = self.reference_vectors.copy()
            
            # Add small random perturbations
            perturbation = np.random.randn(3, self.n_stains) * 0.05
            M_init = M_init + perturbation
            M_init = self._ensure_valid_stain_matrix(M_init)
            
            Y_positive = Y - np.min(Y) + 0.01
            
            try:
                W_init = np.maximum(0.01, pinv(M_init) @ Y_positive).T
                H_init = M_init.T
                
                W_init = np.ascontiguousarray(W_init)
                H_init = np.ascontiguousarray(H_init)
                
                nmf = NMF(n_components=self.n_stains, 
                         init='custom',
                         max_iter=500,
                         alpha_W=0.01, 
                         alpha_H=0.01,
                         l1_ratio=0.5)
                
                C_nmf = nmf.fit_transform(Y_positive.T, W=W_init, H=H_init).T
                M_refined = nmf.components_.T
                
                M_refined = self._ensure_valid_stain_matrix(M_refined)
                is_valid, issues = self.validate_stain_vectors(M_refined)
                
                if is_valid:
                    M_init = M_refined
                else:
                    logger.warning("Hybrid refinement failed validation, keeping reference")
                    
                init_info['nmf_error'] = nmf.reconstruction_err_
                
            except Exception as e:
                logger.warning(f"Hybrid refinement failed: {e}, using reference")
        
        # Final validation and correction
        M_init = self._ensure_valid_stain_matrix(M_init)
        
        # Compute initial concentrations
        C_init = np.maximum(0, pinv(M_init) @ Y)
        C_init = C_init + 0.001
        
        # Final validation
        is_valid, issues = self.validate_stain_vectors(M_init)
        init_info['is_valid'] = is_valid
        init_info['issues'] = issues
        
        return M_init, C_init, init_info
    
    def adaptive_bksvd(self, Y, M_init, C_init, max_iter=30, tol=1e-4):
        """Adaptive BKSVD with robust safeguards against degenerate solutions."""
        Ns = M_init.shape[1]
        Q = Y.shape[1]
        
        M_hat = self._ensure_valid_stain_matrix(M_init.copy())
        C_hat = C_init.copy()
        
        gamma = np.ones((Ns, Q)) * 0.1
        lambda_param = np.ones(Q) * self.sparsity_weight
        beta_hat = min(self.beta_prior, 20.0)
        
        Sigma_c = [np.eye(Ns) for _ in range(Q)]
        
        M_prev = M_hat.copy()
        beta_history = []
        
        for iteration in range(max_iter):
            # Update lambda adaptively
            for q in range(Q):
                active_mask = C_hat[:, q] > 0.05 * np.max(C_hat[:, q])
                n_active = np.sum(active_mask)
                
                if n_active > 0:
                    lambda_param[q] = self.sparsity_weight * Ns / n_active
                else:
                    lambda_param[q] = self.sparsity_weight
            
            # Update gamma with safeguards
            for q in range(Q):
                for s in range(Ns):
                    term1 = -1 / (2 * lambda_param[q])
                    term2_inner = 1/(4 * lambda_param[q]**2) + \
                                 (C_hat[s, q]**2 + Sigma_c[q][s, s]) / lambda_param[q]
                    
                    if term2_inner > 0:
                        term2 = np.sqrt(term2_inner)
                        gamma_new = max(self.EPS, term1 + term2)
                        
                        threshold = 0.05 if C_hat[s, q] > 0.1 else 0.15
                        if gamma_new < threshold:
                            gamma_new = self.EPS
                        
                        gamma[s, q] = gamma_new
            
            # Update concentrations with regularization
            for q in range(Q):
                M_T_M = M_hat.T @ M_hat
                gamma_inv = np.diag(1.0 / (gamma[:, q] + self.EPS))
                
                try:
                    reg_term = 1e-4 * np.eye(Ns)
                    matrix_to_invert = beta_hat * M_T_M + gamma_inv + reg_term
                    Sigma_c[q] = np.linalg.inv(matrix_to_invert)
                    C_hat[:, q] = beta_hat * Sigma_c[q] @ M_hat.T @ Y[:, q]
                    
                    C_hat[:, q] = np.maximum(0.001, C_hat[:, q])
                    
                except np.linalg.LinAlgError:
                    logger.warning(f"Matrix inversion error at pixel {q}")
            
            # Update stain vectors with robust safeguards
            for s in range(Ns):
                conc_sum = np.sum(C_hat[s, :])
                if conc_sum < 0.01:
                    logger.warning(f"Low concentration sum for stain {s}: {conc_sum}")
                    C_hat[s, :] += 0.01
                    conc_sum = np.sum(C_hat[s, :])
                
                residual = Y.copy()
                for i in range(Ns):
                    if i != s:
                        residual -= np.outer(M_hat[:, i], C_hat[i, :])
                
                weights = C_hat[s, :] / (conc_sum + self.EPS)
                M_s_new = residual @ weights
                
                # Add reference vector influence
                ref_influence = 0.1
                M_s_new = (1 - ref_influence) * M_s_new + ref_influence * self.reference_vectors[:, s]
                
                M_s_new = np.maximum(0, M_s_new)
                
                if np.linalg.norm(M_s_new) < 0.01:
                    logger.warning(f"Near-zero update for stain {s}, keeping previous")
                    continue
                
                # Limit change per iteration
                if iteration > 3:
                    max_change = 0.1
                    change = M_s_new - M_hat[:, s]
                    if np.linalg.norm(change) > max_change:
                        M_s_new = M_hat[:, s] + max_change * change / np.linalg.norm(change)
                
                norm = np.linalg.norm(M_s_new)
                if norm > 0.01:
                    M_hat[:, s] = M_s_new / norm
                else:
                    logger.warning(f"Cannot normalize stain {s}, keeping previous")
            
            M_hat = self._ensure_valid_stain_matrix(M_hat)
            
            # Update beta with stability
            residual = Y - M_hat @ C_hat
            sse = np.sum(residual**2)
            term2 = sum(np.trace(M_hat.T @ M_hat @ Sigma_c[q]) for q in range(Q))
            
            regularization_term = 0.1 * Q
            beta_hat = (3*Q + 2) / (sse + term2 + 2 + regularization_term)
            beta_hat = np.clip(beta_hat, 0.5, 30.0)
            beta_history.append(beta_hat)
            
            # Check convergence
            M_diff = np.linalg.norm(M_hat - M_prev) / (np.linalg.norm(M_prev) + self.EPS)
            
            if M_diff < tol:
                print(f"BKSVD converged after {iteration+1} iterations")
                sys.stdout.flush()
                break
            
            M_prev = M_hat.copy()
            
            # Early stopping
            if beta_hat > 25 and iteration > 8:
                logger.warning(f"Early stopping due to high beta: {beta_hat:.2f}")
                break
        
        M_hat = self._ensure_valid_stain_matrix(M_hat)
        
        return M_hat, C_hat, gamma, beta_hat, beta_history


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def rgb_to_od(img, background=None):
    """Convert RGB to optical density."""
    if background is None:
        background = np.array([255, 255, 255])
    
    img_float = img.astype(float)
    background = background.astype(float)
    
    eps = 1e-8
    img_float = np.maximum(img_float, eps)
    
    od = -np.log10(img_float / background[np.newaxis, np.newaxis, :])
    return od


def od_to_rgb(od, background=None):
    """Convert optical density back to RGB."""
    if background is None:
        background = np.array([255, 255, 255])
    
    rgb = background[np.newaxis, np.newaxis, :] * np.power(10, -od)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def process_single_tile(tile_path, pixels_per_tile=1000):
    """
    Process a single tile and return OD samples.
    
    Args:
        tile_path: Path to the tile image
        pixels_per_tile: Number of pixels to sample from this tile
        
    Returns:
        Tuple of (od_sampled, tile_info) or (None, None) if error
    """
    try:
        img = cv2.imread(str(tile_path))
        if img is None:
            return None, None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a flat representation of the image
        h, w, c = img_rgb.shape
        img_flat = img_rgb.reshape(h*w, c)
        
        # Find pixels that are not white (background removed)
        non_white_mask = ~np.all(img_flat == 255, axis=1)
        non_white_indices = np.where(non_white_mask)[0]
        
        if len(non_white_indices) == 0:
            return None, None
        
        # Convert to optical density (only tissue pixels)
        tissue_pixels = img_flat[non_white_indices]
        od = rgb_to_od(tissue_pixels.reshape(-1, 1, 3)).reshape(-1, 3).T
        
        # Sample pixels
        n_pixels = min(pixels_per_tile, od.shape[1])
        if n_pixels < 10:
            return None, None
            
        indices = np.random.choice(od.shape[1], n_pixels, replace=False)
        od_sampled = od[:, indices]
        
        return od_sampled, {'tile_path': tile_path, 'n_pixels': n_pixels}
        
    except Exception as e:
        logger.warning(f"Error processing {tile_path}: {e}")
        return None, None


def sample_pixels_from_tiles(tile_paths, pixels_per_tile=1000, max_tiles=None, max_processes=None):
    """Sample pixels from multiple tiles for WSI-level analysis."""
    if max_tiles:
        tile_paths = tile_paths[:max_tiles]
    
    if max_processes is None:
        available_cpus = multiprocessing.cpu_count()
        if available_cpus >= 8:
            max_processes = available_cpus - 1
        else:
            max_processes = max(1, available_cpus // 2)
    
    print(f"Sampling from {len(tile_paths)} tiles using {max_processes} CPU cores...")
    sys.stdout.flush()
    
    # Create a partial function with fixed pixels_per_tile parameter
    process_tile_with_params = partial(process_single_tile, pixels_per_tile=pixels_per_tile)
    
    ctx = multiprocessing.get_context('spawn')
    
    with ctx.Pool(processes=max_processes) as pool:
        # Process tiles and collect results
        results = pool.map(process_tile_with_params, tile_paths)
    
    # Filter and combine results
    od_samples = []
    tile_info = []
    
    for od_result, info_result in results:
        if od_result is not None and info_result is not None:
            od_samples.append(od_result)
            tile_info.append(info_result)
    
    if not od_samples:
        raise ValueError("No valid pixels sampled from tiles")
    
    od_combined = np.hstack(od_samples)
    print(f"Total pixels sampled: {od_combined.shape[1]}")
    sys.stdout.flush()
    
    return od_combined, tile_info


def load_reference_stain_vectors(parameters_dir):
    """Load reference stain vectors from text file in Parameters directory."""
    reference_file = os.path.join(parameters_dir, "reference_stain_vectors.txt")
    
    print(f"Looking for reference stain vectors at: {reference_file}")
    sys.stdout.flush()
    
    if not os.path.exists(reference_file):
        print(f"Reference stain vectors file not found: {reference_file}")
        sys.stdout.flush()
        logger.warning(f"Reference stain vectors file not found: {reference_file}")
        return None
    
    try:
        stain_matrix = np.zeros((3, 3))  # 3 RGB channels, 3 stains
        stain_names = []
        
        with open(reference_file, 'r') as f:
            lines = f.readlines()
            
            vector_lines = []
            for line in lines:
                if ':' in line and '[' in line:
                    vector_lines.append(line)
            
            if len(vector_lines) < 3:
                raise ValueError("Not enough stain vectors found in file")
            
            for i, line in enumerate(vector_lines[:3]):
                # Extract the vector part [0.123, 0.456, 0.789]
                parts = line.split('[')
                if len(parts) < 2:
                    continue
                    
                # Extract stain name
                stain_name = parts[0].strip().rstrip(':')
                stain_names.append(stain_name)
                
                # Extract vector values
                vector_str = parts[1].split(']')[0]
                vector_values = [float(val.strip()) for val in vector_str.split(',')]
                
                if len(vector_values) == 3:
                    stain_matrix[:, i] = vector_values
        
        print(f"Successfully loaded reference stain vectors from {reference_file}")
        print("Reference stain vectors:")
        for i, name in enumerate(stain_names):
            vector_str = ", ".join([f"{val:.4f}" for val in stain_matrix[:, i]])
            print(f"  {name}: [{vector_str}]")
        sys.stdout.flush()
        
        logger.info(f"Loaded reference stain vectors from {reference_file}")
        
        return stain_matrix
    
    except Exception as e:
        print(f"Error loading reference stain vectors: {str(e)}")
        sys.stdout.flush()
        logger.error(f"Error loading reference stain vectors: {str(e)}")
        return None


def save_stain_vectors(M_final, parameters_dir):
    """Save the final stain vectors to Parameters directory."""
    # Create output directory if it doesn't exist
    os.makedirs(parameters_dir, exist_ok=True)
    
    # Use the same stain names as in reference file
    stain_names = ["Nuclei_HE", "Myelin_Blue", "Microglia_Brown"]
    
    # Save to text file in same format as reference
    stain_vectors_path = os.path.join(parameters_dir, "stain_vectors.txt")
    with open(stain_vectors_path, "w") as f:
        f.write("WSI Stain Vectors:\n")
        f.write("=" * 50 + "\n\n")
        for i, name in enumerate(stain_names):
            vector_str = ", ".join([f"{val:.4f}" for val in M_final[:, i]])
            f.write(f"{name}: [{vector_str}]\n")
    
    print(f"Stain vectors saved to {stain_vectors_path}")
    print("Final stain vectors:")
    for i, name in enumerate(stain_names):
        vector_str = ", ".join([f"{val:.4f}" for val in M_final[:, i]])
        print(f"  {name}: [{vector_str}]")
    sys.stdout.flush()
    
    logger.info(f"Stain vectors saved to {stain_vectors_path}")


def create_visualization(img_rgb, M_final, output_dir, filename="stain_separation"):
    """Create visualization of stain separation and save to Results/EBKSVD"""
    h, w = img_rgb.shape[:2]
    
    od = rgb_to_od(img_rgb)
    od_flat = od.reshape(-1, 3).T
    
    # Calculate concentrations
    try:
        C = np.maximum(0, pinv(M_final, rcond=1e-10) @ od_flat)
    except:
        logger.warning("Using least squares for concentration estimation")
        C = np.maximum(0, np.linalg.lstsq(M_final, od_flat, rcond=None)[0])
    
    C_image = C.reshape(3, h, w)
    
    # Create separated stain images
    separated_stains = []
    stain_names = ["Nuclei_HE", "Myelin_Blue", "Microglia_Brown"]
    
    for i in range(3):
        C_single = np.zeros_like(C)
        C_single[i] = C[i]
        
        od_single = M_final @ C_single
        img_single = od_to_rgb(od_single.T.reshape(h, w, 3))
        separated_stains.append(img_single)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Top row: Original and separated stains
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    for i, (stain, name) in enumerate(zip(separated_stains, stain_names)):
        axes[0, i+1].imshow(stain)
        axes[0, i+1].set_title(name.replace('_', ' '), fontsize=12)
        axes[0, i+1].axis('off')
        
        # Save individual stain images
        stain_filename = os.path.join(output_dir, f"{filename}_{name}.png")
        cv2.imwrite(stain_filename, cv2.cvtColor(stain, cv2.COLOR_RGB2BGR))
    
    # Bottom row: Reconstruction and concentrations
    od_recon = np.zeros_like(od)
    for i in range(3):
        od_recon += rgb_to_od(separated_stains[i])
    img_recon = od_to_rgb(od_recon)
    
    axes[1, 0].imshow(img_recon)
    axes[1, 0].set_title("Reconstructed", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Plot concentrations
    for i, name in enumerate(stain_names):
        conc = C_image[i]
        vmin, vmax = np.percentile(conc, [1, 99])
        im = axes[1, i+1].imshow(conc, cmap='hot', vmin=vmin, vmax=vmax)
        axes[1, i+1].set_title(f"{name.replace('_', ' ')} Conc.", fontsize=12)
        axes[1, i+1].axis('off')
        
        plt.colorbar(im, ax=axes[1, i+1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    visualization_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(visualization_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Stain separation visualization saved to {visualization_path}")
    sys.stdout.flush()


def create_diagnostic_plots(M_estimates, beta_history, init_info, output_dir):
    """Create diagnostic plots for algorithm analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Stain vector evolution
    if len(M_estimates) > 1:
        for stain_idx in range(3):
            for channel_idx in range(3):
                values = [M[:, stain_idx][channel_idx] for M in M_estimates]
                axes[0, stain_idx].plot(values, label=f'Channel {channel_idx}')
        
        stain_names = ["Nuclei_HE", "Myelin_Blue", "Microglia_Brown"]
        for i, name in enumerate(stain_names):
            axes[0, i].set_title(f'{name.replace("_", " ")} Evolution')
            axes[0, i].set_xlabel('Batch')
            axes[0, i].set_ylabel('Value')
            axes[0, i].legend()
            axes[0, i].grid(True)
            axes[0, i].set_ylim(-0.1, 1.1)
    
    # Plot 2: Beta evolution
    axes[1, 0].plot(beta_history, 'b-o')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Beta')
    axes[1, 0].set_title('Beta Evolution')
    axes[1, 0].grid(True)
    axes[1, 0].axhline(y=5, color='g', linestyle='--', alpha=0.5, label='Target zone')
    axes[1, 0].axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='Warning')
    axes[1, 0].axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Max allowed')
    axes[1, 0].legend()
    
    # Plot 3: Initialization quality
    axes[1, 1].text(0.1, 0.9, "Initialization Report", fontsize=14, fontweight='bold')
    y_pos = 0.7
    
    if 'method' in init_info:
        axes[1, 1].text(0.1, y_pos, f"Method: {init_info['method']}", fontsize=12)
        y_pos -= 0.1
    
    if 'nmf_error' in init_info:
        axes[1, 1].text(0.1, y_pos, f"NMF Error: {init_info['nmf_error']:.4f}", fontsize=12)
        y_pos -= 0.1
    
    if 'is_valid' in init_info:
        color = 'green' if init_info['is_valid'] else 'red'
        axes[1, 1].text(0.1, y_pos, f"Validation: {'PASSED' if init_info['is_valid'] else 'FAILED'}", 
                       fontsize=12, color=color)
        y_pos -= 0.1
    
    if 'issues' in init_info and init_info['issues']:
        axes[1, 1].text(0.1, y_pos, "Issues:", fontsize=12)
        y_pos -= 0.1
        for issue in init_info['issues'][:3]:
            axes[1, 1].text(0.1, y_pos, f"  - {issue[:50]}...", fontsize=10)
            y_pos -= 0.1
    
    axes[1, 1].axis('off')
    
    # Plot 4: Final stain matrix
    M_final = M_estimates[-1] if M_estimates else None
    if M_final is not None:
        im = axes[1, 2].imshow(M_final, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        axes[1, 2].set_title('Final Stain Matrix')
        axes[1, 2].set_xlabel('Stain')
        axes[1, 2].set_ylabel('Channel (RGB)')
        axes[1, 2].set_xticks([0, 1, 2])
        axes[1, 2].set_xticklabels(['Nuclei', 'Myelin', 'Microglia'])
        axes[1, 2].set_yticks([0, 1, 2])
        axes[1, 2].set_yticklabels(['R', 'G', 'B'])
        
        # Add values
        for i in range(3):
            for j in range(3):
                text = axes[1, 2].text(j, i, f'{M_final[i, j]:.3f}',
                                     ha="center", va="center", color="white")
        
        plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    diagnostic_path = os.path.join(output_dir, 'diagnostic_plots.png')
    plt.savefig(diagnostic_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Diagnostic plots saved to {diagnostic_path}")
    sys.stdout.flush()


def robust_combine_estimates(M_list):
    """Robustly combine multiple stain matrix estimates."""
    # Use median for robustness
    M_median = np.median(M_list, axis=0)
    
    # Ensure non-negative values
    M_median = np.maximum(0, M_median)
    
    # Normalize each stain vector
    for i in range(M_median.shape[1]):
        norm = np.linalg.norm(M_median[:, i])
        if norm > 1e-10:
            M_median[:, i] /= norm
    
    return M_median


def compute_stain_matrix_variation(M_list):
    """Compute variation across multiple stain matrix estimates."""
    if len(M_list) < 2:
        return float('inf')
    
    M_mean = np.mean(M_list, axis=0)
    variations = [np.linalg.norm(M - M_mean) for M in M_list]
    return np.mean(variations)


def find_test_tile(image_files, test_tile_name):
    """Find the specified test tile in the list of image files."""
    # First try exact match
    for img_path in image_files:
        if os.path.basename(img_path) == test_tile_name:
            return img_path
    
    # If not found, try partial match (without extension)
    test_name_no_ext = os.path.splitext(test_tile_name)[0]
    for img_path in image_files:
        if os.path.splitext(os.path.basename(img_path))[0] == test_name_no_ext:
            return img_path
    
    # If still not found, return None
    return None


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main function for EBKSVD stain vector estimation."""
    parser = argparse.ArgumentParser(description='EBKSVD Stain Vector Estimation for IHC Pipeline')
    parser.add_argument('--data-dir', help='Base data directory')
    parser.add_argument('--output-dir', help='Base output directory')
    parser.add_argument('--parameters-dir', help='Parameters directory')
    parser.add_argument('--input', help='Input folder containing image files (for standalone use)')
    parser.add_argument('--output', help='Output folder for results (for standalone use)')
    parser.add_argument('--pixels-per-tile', type=int, default=1000, help='Pixels to sample per tile')
    parser.add_argument('--max-tiles', type=int, help='Maximum tiles to process')
    parser.add_argument('--batch-size', type=int, default=3000, help='BKSVD batch size')
    parser.add_argument('--n-batches', type=int, default=30, help='Number of BKSVD batches')
    parser.add_argument('--init-method', choices=['nmf', 'reference', 'hybrid'], 
                       default='hybrid', help='Initialization method')
    parser.add_argument('--test-tile', default='corrected_tile_r15_c18.tif', 
                       help='Test tile to use for creating validation visualizations (default: tile_r15_c18.tif)')
    
    args = parser.parse_args()
    
    # Determine input/output paths
    if args.data_dir and args.output_dir and args.parameters_dir:
        # Called from GUI
        input_folder = os.path.join(args.data_dir, "Tiles-Medium-L-Channel-Normalized-BG-Removed-Illumination-Corrected")
        results_folder = os.path.join(args.output_dir, "EBKSVD")
        parameters_dir = args.parameters_dir
        
        # Set up log directory
        base_dir = os.path.dirname(args.data_dir)
        log_dir = os.path.join(base_dir, "Logs")
    elif args.input and args.output:
        # Called standalone
        input_folder = args.input
        results_folder = args.output
        parameters_dir = args.parameters_dir if args.parameters_dir else os.path.join(os.path.dirname(args.output), "Parameters")
        log_dir = "Logs"
    else:
        print("ERROR: Either provide --data-dir, --output-dir, and --parameters-dir OR provide --input and --output")
        sys.stdout.flush()
        return
    
    # Create output directories
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(parameters_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging to file
    log_file = os.path.join(log_dir, "ebksvd.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log startup information
    print("Starting EBKSVD Stain Vector Estimation")
    print(f"Input folder: {input_folder}")
    print(f"Results folder: {results_folder}")
    print(f"Parameters folder: {parameters_dir}")
    print(f"Test tile: {args.test_tile}")
    print(f"Log file: {log_file}")
    sys.stdout.flush()
    
    logger.info("Starting EBKSVD Stain Vector Estimation")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Results folder: {results_folder}")
    logger.info(f"Parameters folder: {parameters_dir}")
    logger.info(f"Test tile: {args.test_tile}")
    
    # Load reference stain vectors
    reference_stain_matrix = load_reference_stain_vectors(parameters_dir)
    
    if reference_stain_matrix is None:
        error_msg = "No reference-stain-vectors found for algorithm initialization, please run \"Import Reference Stain Vectors\" first."
        print(error_msg)
        sys.stdout.flush()
        logger.error(error_msg)
        return
    
    # Find all image files in input directory
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        sys.stdout.flush()
        logger.error(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} image files in {input_folder}")
    sys.stdout.flush()
    
    try:
        # Sample pixels from tiles
        print("Sampling pixels from tiles...")
        sys.stdout.flush()
        
        od_samples, tile_info = sample_pixels_from_tiles(
            [Path(f) for f in image_files], 
            args.pixels_per_tile, 
            args.max_tiles
        )
        
        # Initialize algorithm with reference vectors
        bksvd = ImprovedThreeStainBKSVD(
            n_stains=3, 
            sparsity_weight=0.05,
            beta_prior=5.0,
            reference_vectors=reference_stain_matrix
        )
        
        # Multi-batch processing
        M_estimates = []
        all_beta_history = []
        init_info = None
        
        print(f"Running {args.n_batches} BKSVD batches...")
        sys.stdout.flush()
        
        for batch_idx in range(args.n_batches):
            if batch_idx % 5 == 0:
                print(f"Processing batch {batch_idx+1}/{args.n_batches}")
                sys.stdout.flush()
            
            n_pixels = min(args.batch_size, od_samples.shape[1])
            indices = np.random.choice(od_samples.shape[1], n_pixels, replace=False)
            od_batch = od_samples[:, indices]
            
            # Initialize only on first batch
            if batch_idx == 0:
                M_init, C_init, init_info = bksvd.smart_initialization(
                    od_batch, method=args.init_method
                )
                logger.info(f"Initialization method: {args.init_method}")
                if init_info['is_valid']:
                    logger.info("Initialization validation PASSED")
                else:
                    logger.warning("Initialization validation failed")
            else:
                M_init = M_estimates[-1].copy()
                C_init = np.maximum(0, pinv(M_init) @ od_batch) + 0.001
            
            # Run BKSVD
            M_batch, C_batch, gamma, beta, beta_history = bksvd.adaptive_bksvd(
                od_batch, M_init, C_init, max_iter=25
            )
            
            M_estimates.append(M_batch)
            all_beta_history.extend(beta_history)
            
            # Check convergence
            if len(M_estimates) >= 5:
                variation = compute_stain_matrix_variation(M_estimates[-5:])
                if variation < 0.08:
                    print(f"Converged after {batch_idx + 1} batches")
                    sys.stdout.flush()
                    break
        
        # Combine estimates
        print("Combining stain matrix estimates...")
        sys.stdout.flush()
        M_final = robust_combine_estimates(M_estimates)
        
        # Final validation
        is_valid, issues = bksvd.validate_stain_vectors(M_final)
        if not is_valid:
            logger.warning("Final stain vectors have issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            print("Final stain vectors validation PASSED")
            sys.stdout.flush()
        
        # Save final stain vectors
        save_stain_vectors(M_final, parameters_dir)
        
        # Create diagnostic plots
        print("Creating diagnostic plots...")
        sys.stdout.flush()
        create_diagnostic_plots(M_estimates, all_beta_history, init_info, results_folder)
        
        # Apply to test tile for visualization
        if image_files:
            print("Creating stain separation visualization...")
            sys.stdout.flush()
            
            # Find the specified test tile
            test_tile_path = find_test_tile(image_files, args.test_tile)
            
            if test_tile_path is None:
                print(f"Warning: Test tile '{args.test_tile}' not found. Using first available tile.")
                sys.stdout.flush()
                logger.warning(f"Test tile '{args.test_tile}' not found. Using first available tile.")
                test_tile_path = image_files[0]
            else:
                print(f"Using test tile: {os.path.basename(test_tile_path)}")
                sys.stdout.flush()
                logger.info(f"Using test tile: {os.path.basename(test_tile_path)}")
            
            img = cv2.imread(test_tile_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                create_visualization(img_rgb, M_final, results_folder, "stain_separation")
        
        print(f"\nResults saved to: {results_folder}")
        print(f"Stain vectors saved to: {parameters_dir}")
        sys.stdout.flush()
        
        logger.info(f"Results saved to: {results_folder}")
        logger.info(f"Stain vectors saved to: {parameters_dir}")
        logger.info("EBKSVD completed successfully")
        
    except Exception as e:
        print(f"Error in EBKSVD processing: {str(e)}")
        sys.stdout.flush()
        logger.error(f"Error in EBKSVD processing: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    # For Windows compatibility
    multiprocessing.freeze_support()
    # Configure basic console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    main()
