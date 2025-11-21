# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 17:12:05 2025

@author: TRUE Lab
"""

# my_visualization.py

import numpy as np
import matplotlib.pyplot as plt
import imageio

def make_sinogram_video(results, RF, RFsum, n_avg, video_name="sinogram_results.mp4", folder_name=None, fps=5):
    """
    Create a video of noisy, ground truth, and predicted sinograms.
    """
    import os
    
    base_dir = os.getcwd()
    if folder_name is None:
        folder_name = base_dir
    
    os.makedirs(folder_name, exist_ok=True)
    out_path = os.path.join(folder_name, video_name)
    
    writer = imageio.get_writer(out_path, fps=fps)
    Tot_frames = results.shape[2]
    
    for frame_ind in range(Tot_frames):
        reshaped_frame = results[:, :, frame_ind]
        noisy_frame = np.mean(RF[:, :, frame_ind * n_avg:(frame_ind + 1) * n_avg], axis=2)
        
        r_denoised_rf = np.corrcoef(RFsum.flatten(), reshaped_frame.flatten())[0, 1]
        print(f"Correlation coefficient (denoised vs RFsum) for frame {frame_ind}: {r_denoised_rf:.4f}")
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        
        im0 = axes[0].imshow(noisy_frame, aspect='auto', cmap='gray')
        axes[0].set_title(f"Noisy Sinogram (frame {frame_ind})")
        fig.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(RFsum, aspect='auto', cmap='gray')
        axes[1].set_title("Ground Truth Sinogram (RFsum)")
        fig.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(reshaped_frame, aspect='auto', cmap='gray')
        axes[2].set_title(f"Predicted Sinogram (r = {r_denoised_rf:.4f})")
        fig.colorbar(im2, ax=axes[2])
        
        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())
        image = image[:, :, :3]
        writer.append_data(image)
        
        plt.close(fig)
    
    writer.close()
    print(f"✅ Video saved successfully!")
    print(f"📂 Location: {folder_name}")
    print(f"🎥 File: {os.path.basename(out_path)}")

def plot_mip(volume, h=None, cmap="hot"):
    """
    Plot maximum intensity projections (MIPs) of a 3D volume along x, y, z axes.
    """
    
    if h is not None:
        if np.isscalar(h):
            dx = dy = dz = float(h)
        else:
            dx, dy, dz = h
    else:
        dx = dy = dz = 1.0
    
    Nx, Ny, Nz = volume.shape
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dy
    z = (np.arange(Nz) - Nz // 2) * dz
    
    proj_x = np.max(volume, axis=0)
    proj_y = np.max(volume, axis=1)
    proj_z = np.max(volume, axis=2)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axs[0].imshow(proj_x.T, extent=[y.min(), y.max(), z.min(), z.max()],
                        origin="lower", cmap=cmap, aspect="equal")
    axs[0].set_title("Max over X-axis")
    axs[0].set_xlabel("y [m]" if h else "y [voxels]")
    axs[0].set_ylabel("z [m]" if h else "z [voxels]")
    plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
    
    im2 = axs[1].imshow(proj_y.T, extent=[x.min(), x.max(), z.min(), z.max()],
                        origin="lower", cmap=cmap, aspect="equal")
    axs[1].set_title("Max over Y-axis")
    axs[1].set_xlabel("x [m]" if h else "x [voxels]")
    axs[1].set_ylabel("z [m]" if h else "z [voxels]")
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)
    
    im3 = axs[2].imshow(proj_z.T, extent=[x.min(), x.max(), y.min(), y.max()],
                        origin="lower", cmap=cmap, aspect="equal")
    axs[2].set_title("Max over Z-axis")
    axs[2].set_xlabel("x [m]" if h else "x [voxels]")
    axs[2].set_ylabel("y [m]" if h else "y [voxels]")
    plt.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    return fig, axs

def save_volume_video(frames, movie_name="volume_video.mp4", folder_name=None, h=None, fps=10, nticks=5):
    """
    Stub for pyvista volume video - not available in cloud.
    """
    print("⚠️ 3D volume video generation skipped (PyVista not available in Streamlit Cloud)")
    print(f"   Would have created: {movie_name}")
    pass
