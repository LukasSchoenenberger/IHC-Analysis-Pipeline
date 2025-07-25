import numpy as np
import nibabel as nib
from scipy import ndimage
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from queue import PriorityQueue
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox

def select_input_files():
    """
    Opens a GUI file dialog to select multiple segmentation files
    """
    print("Creating file selection dialog...")
    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.lift()  # Bring to front
    root.attributes('-topmost', True)  # Keep on top
    
    try:
        file_paths = filedialog.askopenfilenames(
            title="Select Segmentation Files to Split",
            filetypes=[
                ("NIfTI files", "*.nii *.nii.gz"),
                ("NIfTI compressed", "*.nii.gz"),
                ("NIfTI uncompressed", "*.nii"),
                ("All files", "*.*")
            ],
            initialdir=Path.cwd(),
            parent=root
        )
        
        result = list(file_paths) if file_paths else []
        print(f"File dialog closed. Selected {len(result)} files.")
        
    except Exception as e:
        print(f"Error in file dialog: {e}")
        result = []
    
    finally:
        root.destroy()  # Clean up the root window
        root.quit()     # Ensure tkinter exits properly
    
    return result

def get_neighbors(point, shape):
    """Get 8-connected neighbors of a point"""
    y, x = point
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                neighbors.append((ny, nx))
    return neighbors

def grow_region_from_seed(component, seed, available_mask, target_size=500):
    """
    Grow a region from a seed point using controlled growth with 8-connectivity
    """
    region = np.zeros_like(component, dtype=bool)
    seed = tuple(seed)
    
    if not available_mask[seed]:
        return region, available_mask
    
    # Initialize region with seed point
    region[seed] = True
    available_mask[seed] = False
    pixels_added = 1
    
    # Priority queue for boundary pixels
    # Format: (distance_from_seed, distance_from_boundary, point)
    pq = PriorityQueue()
    
    # Add initial neighbors to queue
    for neighbor in get_neighbors(seed, component.shape):
        if available_mask[neighbor]:
            distance = np.sqrt(((neighbor[0] - seed[0])**2 + (neighbor[1] - seed[1])**2))
            pq.put((distance, 0, neighbor))
    
    while not pq.empty() and pixels_added < target_size:
        _, _, current = pq.get()
        
        if not available_mask[current]:
            continue
            
        # Add pixel to region
        region[current] = True
        available_mask[current] = False
        pixels_added += 1
        
        # Add new neighbors to queue
        for neighbor in get_neighbors(current, component.shape):
            if available_mask[neighbor] and component[neighbor]:
                # Calculate distances
                dist_from_seed = np.sqrt(((neighbor[0] - seed[0])**2 + (neighbor[1] - seed[1])**2))
                dist_from_boundary = 0  # Could be modified for different growth patterns
                
                pq.put((dist_from_seed, dist_from_boundary, neighbor))
    
    return region, available_mask

def find_nearest_valid_point(point, component):
    """Find the nearest point within the component"""
    if component[tuple(point)]:
        return point
        
    valid_points = np.argwhere(component)
    distances = np.sqrt(np.sum((valid_points - point) ** 2, axis=1))
    nearest_idx = np.argmin(distances)
    return valid_points[nearest_idx]
    
    
def visualize_steps(component, potential_seeds, seed_points, initial_regions, final_regions, title, output_path):
    """Visualize all steps of the segmentation process"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # 1. Original component
    axes[0,0].imshow(component, cmap='binary')
    axes[0,0].set_title('Original Component')
    axes[0,0].axis('off')
    
    # 2. Handle seed points visualization
    axes[0,1].imshow(component, cmap='binary')
    if seed_points is not None and len(seed_points) > 0:
        if potential_seeds is not None and len(potential_seeds) > 0:
            axes[0,0].plot(potential_seeds[:, 1], potential_seeds[:, 0], 'r.', markersize=2)
            axes[0,0].set_title('Potential Seed Points\n(top 25% distance transform)')
            
        axes[0,1].plot(seed_points[:, 1], seed_points[:, 0], 'r.', markersize=10)
        axes[0,1].set_title('Selected Seed Points\n(k-means cluster centers)')
    else:
        axes[0,1].set_title('Single Region\n(no splitting required)')
    axes[0,1].axis('off')
    
    # 3 & 4. Handle regions visualization
    if initial_regions is not None and final_regions is not None:
        # Initial regions
        n_regions_initial = len(np.unique(initial_regions)) - 1
        colors_initial = plt.cm.tab20(np.linspace(0, 1, max(n_regions_initial+1, 2)))
        colors_initial[0] = [0, 0, 0, 1]
        cmap_initial = mcolors.ListedColormap(colors_initial)
        
        axes[1,0].imshow(initial_regions, cmap=cmap_initial)
        axes[1,0].set_title('Initial Regions\n(after 500-pixel growth)')
        
        # Final regions
        n_regions_final = len(np.unique(final_regions)) - 1
        colors_final = plt.cm.tab20(np.linspace(0, 1, max(n_regions_final+1, 2)))
        colors_final[0] = [0, 0, 0, 1]
        cmap_final = mcolors.ListedColormap(colors_final)
        
        im = axes[1,1].imshow(final_regions, cmap=cmap_final)
        axes[1,1].set_title('Final Regions\n(after assigning remaining pixels)')
    else:
        # For unsplit components, show the original in both bottom panels
        axes[1,0].imshow(component, cmap='binary')
        axes[1,0].set_title('Single Region')
        axes[1,1].imshow(component, cmap='binary')
        axes[1,1].set_title('Final Region')
    
    axes[1,0].axis('off')
    axes[1,1].axis('off')
    
    # Add region size information
    if final_regions is not None:
        region_sizes = [np.count_nonzero(final_regions == i) for i in range(1, len(np.unique(final_regions)))]
    else:
        region_sizes = [np.count_nonzero(component)]
        
    if region_sizes:
        mean_size = np.mean(region_sizes)
        std_size = np.std(region_sizes)
        cv = std_size / mean_size if mean_size > 0 else 0
        
        info_text = f'Region sizes:\n'
        for i, size in enumerate(region_sizes, 1):
            info_text += f'Region {i}: {size} pixels\n'
        info_text += f'\nMean: {mean_size:.0f}\n'
        info_text += f'Std: {std_size:.0f}\n'
        info_text += f'CV: {cv:.2f}'
        
        plt.figtext(1.02, 0.5, info_text, fontsize=8, va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_component(component, current_label):
    """Process a single connected component with visualization steps"""
    # Count pixels and determine number of seed points
    n_pixels = np.count_nonzero(component)
    n_seeds = max(1, n_pixels // 500)
    
    # If only one seed needed, return component as is
    if n_seeds == 1:
        return component * current_label, None, None, None, component * current_label
    
    # Compute distance transform for this component
    distance = ndimage.distance_transform_edt(component)
    
    # Get coordinates of all non-zero pixels
    coords = np.argwhere(component)
    distances = distance[component]
    
    # Keep top 25% of pixels based on distance values as potential seeds
    threshold = np.percentile(distances, 25)
    potential_seeds = coords[distances >= threshold]
    
    if len(potential_seeds) < n_seeds:
        return component * current_label, None, None, None, component * current_label
    
    # Apply k-means to cluster potential seed points
    kmeans = KMeans(n_clusters=n_seeds, n_init=10, random_state=42)
    kmeans.fit(potential_seeds)
    seed_points = kmeans.cluster_centers_.astype(int)
    
    # Ensure all seed points are within the component
    valid_seed_points = np.array([
        find_nearest_valid_point(seed, component)
        for seed in seed_points
    ])
    
    # Initialize result arrays
    initial_regions = np.zeros_like(component, dtype=int)
    final_regions = np.zeros_like(component, dtype=int)
    available_mask = component.copy()
    
    # First phase: Controlled region growing
    for i, seed in enumerate(valid_seed_points):
        region, available_mask = grow_region_from_seed(
            component,
            seed,
            available_mask,
            target_size=500
        )
        
        if np.any(region):
            initial_regions[region] = current_label + i
            final_regions[region] = current_label + i
    
    # Second phase: Assign remaining pixels to nearest adjacent region
    unassigned = available_mask & component
    if np.any(unassigned):
        while np.any(unassigned):
            # Find all pixels adjacent to existing regions
            boundary_pixels = np.zeros_like(unassigned, dtype=bool)
            assigned = final_regions > 0
            
            # Find unassigned pixels that are adjacent to assigned regions
            for y, x in np.argwhere(unassigned):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < component.shape[0] and 
                            0 <= nx < component.shape[1] and 
                            assigned[ny, nx]):
                            boundary_pixels[y, x] = True
                            break
                    if boundary_pixels[y, x]:
                        break
            
            if not np.any(boundary_pixels):
                break
                
            # Assign each boundary pixel to the most common region among its neighbors
            for y, x in np.argwhere(boundary_pixels):
                neighbor_labels = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < component.shape[0] and 
                            0 <= nx < component.shape[1] and 
                            final_regions[ny, nx] > 0):
                            neighbor_labels.append(final_regions[ny, nx])
                
                if neighbor_labels:
                    # Assign to the most common neighboring region
                    from collections import Counter
                    most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
                    final_regions[y, x] = most_common_label
                    unassigned[y, x] = False
    
    return final_regions, potential_seeds, valid_seed_points, initial_regions, final_regions

def find_best_slice(data):
    """
    Find the best slice to process by examining all three orientations 
    and selecting the one with the most content.
    """
    # Check content in all three possible orientations
    content_per_slice = []
    
    # For each axis (0, 1, 2)
    for axis in range(3):
        # Compute non-zero pixel count for each slice along this axis
        if axis == 0:
            # Axis 0 - original orientation
            counts = [np.count_nonzero(data[i]) for i in range(data.shape[0])]
        elif axis == 1:
            # Axis 1 - sagittal plane
            counts = [np.count_nonzero(data[:, i]) for i in range(data.shape[1])]
        else:
            # Axis 2 - coronal plane
            counts = [np.count_nonzero(data[:, :, i]) for i in range(data.shape[2])]
            
        if counts:
            max_count = max(counts)
            max_idx = counts.index(max_count)
            content_per_slice.append((axis, max_idx, max_count))
    
    # Sort by content count (descending)
    content_per_slice.sort(key=lambda x: x[2], reverse=True)
    
    if not content_per_slice:
        return None, None
        
    # Return the axis and slice index with the most content
    best_axis, best_slice_idx, _ = content_per_slice[0]
    
    return best_axis, best_slice_idx

def process_selected_files(file_paths):
    """Process the selected segmentation files"""
    if not file_paths:
        print("No files selected.")
        return False
    
    # Always save to Splitted_annotation_masks in current working directory
    output_folder = Path.cwd() / "Splitted_annotation_masks"
    output_folder.mkdir(exist_ok=True)
    
    vis_folder = output_folder / "visualizations"
    vis_folder.mkdir(exist_ok=True)
    
    steps_folder = output_folder / "step_visualizations"
    steps_folder.mkdir(exist_ok=True)
    
    print(f"Output folder: {output_folder}")
    print(f"Found {len(file_paths)} files to process")
    
    for file_path in file_paths:
        file_path = Path(file_path)
        print(f"\nProcessing {file_path.name}...")
        
        try:
            img = nib.load(str(file_path))
            data = img.get_fdata()
            
            # Find the best slice by examining all three orientations
            best_axis, best_slice_idx = find_best_slice(data)
            
            if best_axis is None:
                print(f"Skipping {file_path.name} - no content found")
                continue
                
            # Extract the best 2D slice based on the identified orientation
            if best_axis == 0:
                # Original orientation (axial)
                mask = data[best_slice_idx].astype(bool)
                print(f"Using axial slice {best_slice_idx} (max content)")
            elif best_axis == 1:
                # Sagittal orientation
                mask = data[:, best_slice_idx].astype(bool)
                print(f"Using sagittal slice {best_slice_idx} (max content)")
            else:
                # Coronal orientation
                mask = data[:, :, best_slice_idx].astype(bool)
                print(f"Using coronal slice {best_slice_idx} (max content)")
            
            # Process each connected component
            labeled_components, n_components = ndimage.label(mask)
            result = np.zeros_like(data)
            current_label = 1
            
            print(f"Found {n_components} connected components")
            
            for comp_idx in range(1, n_components + 1):
                print(f"Processing component {comp_idx}/{n_components}")
                component = labeled_components == comp_idx
                component_size = np.count_nonzero(component)
                print(f"Component size: {component_size} pixels")
                
                processed_comp, potential_seeds, seed_points, initial_regions, final_regions = process_component(
                    component, 
                    current_label
                )
                
                # Create visualization for every component
                steps_output_path = steps_folder / f"{file_path.stem}_component{comp_idx}_steps.png"
                visualize_steps(
                    component,
                    potential_seeds,
                    seed_points,
                    initial_regions,
                    final_regions,
                    f"Processing Steps for Component {comp_idx}",
                    steps_output_path
                )
                
                # Place the processed component back in the right orientation
                if best_axis == 0:
                    # Original orientation (axial)
                    result[best_slice_idx][component] = processed_comp[component]
                elif best_axis == 1:
                    # Sagittal orientation
                    result[:, best_slice_idx][component] = processed_comp[component]
                else:
                    # Coronal orientation
                    result[:, :, best_slice_idx][component] = processed_comp[component]
                
                current_label = result.max() + 1
            
            # Save final result
            new_img = nib.Nifti1Image(result, img.affine)
            output_path = output_folder / f"{file_path.stem}_kmeans_split.nii.gz"
            nib.save(new_img, str(output_path))
            print(f"Saved split segmentation to {output_path}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}:")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return True

if __name__ == "__main__":
    try:
        print("Starting segmentation splitter...")
        print("Opening file selection dialog...")
        selected_files = select_input_files()
        
        if selected_files:
            print(f"Selected {len(selected_files)} files:")
            for file_path in selected_files:
                print(f"  - {Path(file_path).name}")
            
            success = process_selected_files(selected_files)
            if success:
                print("\nProcessing completed successfully!")
                print(f"Results saved to: {Path.cwd() / 'Splitted_annotation_masks'}")
            else:
                print("Processing failed!")
        else:
            print("No files selected. Exiting.")
        
        print("Segmentation splitter finished.")
        
    except Exception as e:
        print(f"Error in segmentation splitter: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure we exit cleanly
        import sys
        sys.exit(0)


def get_neighbors(point, shape):
    """Get 8-connected neighbors of a point"""
    y, x = point
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                neighbors.append((ny, nx))
    return neighbors

def grow_region_from_seed(component, seed, available_mask, target_size=500):
    """
    Grow a region from a seed point using controlled growth with 8-connectivity
    """
    region = np.zeros_like(component, dtype=bool)
    seed = tuple(seed)
    
    if not available_mask[seed]:
        return region, available_mask
    
    # Initialize region with seed point
    region[seed] = True
    available_mask[seed] = False
    pixels_added = 1
    
    # Priority queue for boundary pixels
    # Format: (distance_from_seed, distance_from_boundary, point)
    pq = PriorityQueue()
    
    # Add initial neighbors to queue
    for neighbor in get_neighbors(seed, component.shape):
        if available_mask[neighbor]:
            distance = np.sqrt(((neighbor[0] - seed[0])**2 + (neighbor[1] - seed[1])**2))
            pq.put((distance, 0, neighbor))
    
    while not pq.empty() and pixels_added < target_size:
        _, _, current = pq.get()
        
        if not available_mask[current]:
            continue
            
        # Add pixel to region
        region[current] = True
        available_mask[current] = False
        pixels_added += 1
        
        # Add new neighbors to queue
        for neighbor in get_neighbors(current, component.shape):
            if available_mask[neighbor] and component[neighbor]:
                # Calculate distances
                dist_from_seed = np.sqrt(((neighbor[0] - seed[0])**2 + (neighbor[1] - seed[1])**2))
                dist_from_boundary = 0  # Could be modified for different growth patterns
                
                pq.put((dist_from_seed, dist_from_boundary, neighbor))
    
    return region, available_mask

def find_nearest_valid_point(point, component):
    """Find the nearest point within the component"""
    if component[tuple(point)]:
        return point
        
    valid_points = np.argwhere(component)
    distances = np.sqrt(np.sum((valid_points - point) ** 2, axis=1))
    nearest_idx = np.argmin(distances)
    return valid_points[nearest_idx]
    
    
def visualize_steps(component, potential_seeds, seed_points, initial_regions, final_regions, title, output_path):
    """Visualize all steps of the segmentation process"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(title, fontsize=16, y=0.95)
    
    # 1. Original component
    axes[0,0].imshow(component, cmap='binary')
    axes[0,0].set_title('Original Component')
    axes[0,0].axis('off')
    
    # 2. Handle seed points visualization
    axes[0,1].imshow(component, cmap='binary')
    if seed_points is not None and len(seed_points) > 0:
        if potential_seeds is not None and len(potential_seeds) > 0:
            axes[0,0].plot(potential_seeds[:, 1], potential_seeds[:, 0], 'r.', markersize=2)
            axes[0,0].set_title('Potential Seed Points\n(top 25% distance transform)')
            
        axes[0,1].plot(seed_points[:, 1], seed_points[:, 0], 'r.', markersize=10)
        axes[0,1].set_title('Selected Seed Points\n(k-means cluster centers)')
    else:
        axes[0,1].set_title('Single Region\n(no splitting required)')
    axes[0,1].axis('off')
    
    # 3 & 4. Handle regions visualization
    if initial_regions is not None and final_regions is not None:
        # Initial regions
        n_regions_initial = len(np.unique(initial_regions)) - 1
        colors_initial = plt.cm.tab20(np.linspace(0, 1, max(n_regions_initial+1, 2)))
        colors_initial[0] = [0, 0, 0, 1]
        cmap_initial = mcolors.ListedColormap(colors_initial)
        
        axes[1,0].imshow(initial_regions, cmap=cmap_initial)
        axes[1,0].set_title('Initial Regions\n(after 500-pixel growth)')
        
        # Final regions
        n_regions_final = len(np.unique(final_regions)) - 1
        colors_final = plt.cm.tab20(np.linspace(0, 1, max(n_regions_final+1, 2)))
        colors_final[0] = [0, 0, 0, 1]
        cmap_final = mcolors.ListedColormap(colors_final)
        
        im = axes[1,1].imshow(final_regions, cmap=cmap_final)
        axes[1,1].set_title('Final Regions\n(after assigning remaining pixels)')
    else:
        # For unsplit components, show the original in both bottom panels
        axes[1,0].imshow(component, cmap='binary')
        axes[1,0].set_title('Single Region')
        axes[1,1].imshow(component, cmap='binary')
        axes[1,1].set_title('Final Region')
    
    axes[1,0].axis('off')
    axes[1,1].axis('off')
    
    # Add region size information
    if final_regions is not None:
        region_sizes = [np.count_nonzero(final_regions == i) for i in range(1, len(np.unique(final_regions)))]
    else:
        region_sizes = [np.count_nonzero(component)]
        
    if region_sizes:
        mean_size = np.mean(region_sizes)
        std_size = np.std(region_sizes)
        cv = std_size / mean_size if mean_size > 0 else 0
        
        info_text = f'Region sizes:\n'
        for i, size in enumerate(region_sizes, 1):
            info_text += f'Region {i}: {size} pixels\n'
        info_text += f'\nMean: {mean_size:.0f}\n'
        info_text += f'Std: {std_size:.0f}\n'
        info_text += f'CV: {cv:.2f}'
        
        plt.figtext(1.02, 0.5, info_text, fontsize=8, va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_component(component, current_label):
    """Process a single connected component with visualization steps"""
    # Count pixels and determine number of seed points
    n_pixels = np.count_nonzero(component)
    n_seeds = max(1, n_pixels // 500)
    
    # If only one seed needed, return component as is
    if n_seeds == 1:
        return component * current_label, None, None, None, component * current_label
    
    # Compute distance transform for this component
    distance = ndimage.distance_transform_edt(component)
    
    # Get coordinates of all non-zero pixels
    coords = np.argwhere(component)
    distances = distance[component]
    
    # Keep top 25% of pixels based on distance values as potential seeds
    threshold = np.percentile(distances, 25)
    potential_seeds = coords[distances >= threshold]
    
    if len(potential_seeds) < n_seeds:
        return component * current_label, None, None, None, component * current_label
    
    # Apply k-means to cluster potential seed points
    kmeans = KMeans(n_clusters=n_seeds, n_init=10, random_state=42)
    kmeans.fit(potential_seeds)
    seed_points = kmeans.cluster_centers_.astype(int)
    
    # Ensure all seed points are within the component
    valid_seed_points = np.array([
        find_nearest_valid_point(seed, component)
        for seed in seed_points
    ])
    
    # Initialize result arrays
    initial_regions = np.zeros_like(component, dtype=int)
    final_regions = np.zeros_like(component, dtype=int)
    available_mask = component.copy()
    
    # First phase: Controlled region growing
    for i, seed in enumerate(valid_seed_points):
        region, available_mask = grow_region_from_seed(
            component,
            seed,
            available_mask,
            target_size=500
        )
        
        if np.any(region):
            initial_regions[region] = current_label + i
            final_regions[region] = current_label + i
    
    # Second phase: Assign remaining pixels to nearest adjacent region
    unassigned = available_mask & component
    if np.any(unassigned):
        while np.any(unassigned):
            # Find all pixels adjacent to existing regions
            boundary_pixels = np.zeros_like(unassigned, dtype=bool)
            assigned = final_regions > 0
            
            # Find unassigned pixels that are adjacent to assigned regions
            for y, x in np.argwhere(unassigned):
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < component.shape[0] and 
                            0 <= nx < component.shape[1] and 
                            assigned[ny, nx]):
                            boundary_pixels[y, x] = True
                            break
                    if boundary_pixels[y, x]:
                        break
            
            if not np.any(boundary_pixels):
                break
                
            # Assign each boundary pixel to the most common region among its neighbors
            for y, x in np.argwhere(boundary_pixels):
                neighbor_labels = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < component.shape[0] and 
                            0 <= nx < component.shape[1] and 
                            final_regions[ny, nx] > 0):
                            neighbor_labels.append(final_regions[ny, nx])
                
                if neighbor_labels:
                    # Assign to the most common neighboring region
                    from collections import Counter
                    most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
                    final_regions[y, x] = most_common_label
                    unassigned[y, x] = False
    
    return final_regions, potential_seeds, valid_seed_points, initial_regions, final_regions

def find_best_slice(data):
    """
    Find the best slice to process by examining all three orientations 
    and selecting the one with the most content.
    """
    # Check content in all three possible orientations
    content_per_slice = []
    
    # For each axis (0, 1, 2)
    for axis in range(3):
        # Compute non-zero pixel count for each slice along this axis
        if axis == 0:
            # Axis 0 - original orientation
            counts = [np.count_nonzero(data[i]) for i in range(data.shape[0])]
        elif axis == 1:
            # Axis 1 - sagittal plane
            counts = [np.count_nonzero(data[:, i]) for i in range(data.shape[1])]
        else:
            # Axis 2 - coronal plane
            counts = [np.count_nonzero(data[:, :, i]) for i in range(data.shape[2])]
            
        if counts:
            max_count = max(counts)
            max_idx = counts.index(max_count)
            content_per_slice.append((axis, max_idx, max_count))
    
    # Sort by content count (descending)
    content_per_slice.sort(key=lambda x: x[2], reverse=True)
    
    if not content_per_slice:
        return None, None
        
    # Return the axis and slice index with the most content
    best_axis, best_slice_idx, _ = content_per_slice[0]
    
    return best_axis, best_slice_idx

def process_selected_files(file_paths):
    """Process the selected segmentation files"""
    if not file_paths:
        print("No files selected.")
        return False
    
    # Always save to Splitted_annotation_masks in current working directory
    output_folder = Path.cwd() / "Splitted_annotation_masks"
    output_folder.mkdir(exist_ok=True)
    
    vis_folder = output_folder / "visualizations"
    vis_folder.mkdir(exist_ok=True)
    
    steps_folder = output_folder / "step_visualizations"
    steps_folder.mkdir(exist_ok=True)
    
    print(f"Output folder: {output_folder}")
    print(f"Found {len(file_paths)} files to process")
    
    for file_path in file_paths:
        file_path = Path(file_path)
        print(f"\nProcessing {file_path.name}...")
        
        try:
            img = nib.load(str(file_path))
            data = img.get_fdata()
            
            # Find the best slice by examining all three orientations
            best_axis, best_slice_idx = find_best_slice(data)
            
            if best_axis is None:
                print(f"Skipping {file_path.name} - no content found")
                continue
                
            # Extract the best 2D slice based on the identified orientation
            if best_axis == 0:
                # Original orientation (axial)
                mask = data[best_slice_idx].astype(bool)
                print(f"Using axial slice {best_slice_idx} (max content)")
            elif best_axis == 1:
                # Sagittal orientation
                mask = data[:, best_slice_idx].astype(bool)
                print(f"Using sagittal slice {best_slice_idx} (max content)")
            else:
                # Coronal orientation
                mask = data[:, :, best_slice_idx].astype(bool)
                print(f"Using coronal slice {best_slice_idx} (max content)")
            
            # Process each connected component
            labeled_components, n_components = ndimage.label(mask)
            result = np.zeros_like(data)
            current_label = 1
            
            print(f"Found {n_components} connected components")
            
            for comp_idx in range(1, n_components + 1):
                print(f"Processing component {comp_idx}/{n_components}")
                component = labeled_components == comp_idx
                component_size = np.count_nonzero(component)
                print(f"Component size: {component_size} pixels")
                
                processed_comp, potential_seeds, seed_points, initial_regions, final_regions = process_component(
                    component, 
                    current_label
                )
                
                # Create visualization for every component
                steps_output_path = steps_folder / f"{file_path.stem}_component{comp_idx}_steps.png"
                visualize_steps(
                    component,
                    potential_seeds,
                    seed_points,
                    initial_regions,
                    final_regions,
                    f"Processing Steps for Component {comp_idx}",
                    steps_output_path
                )
                
                # Place the processed component back in the right orientation
                if best_axis == 0:
                    # Original orientation (axial)
                    result[best_slice_idx][component] = processed_comp[component]
                elif best_axis == 1:
                    # Sagittal orientation
                    result[:, best_slice_idx][component] = processed_comp[component]
                else:
                    # Coronal orientation
                    result[:, :, best_slice_idx][component] = processed_comp[component]
                
                current_label = result.max() + 1
            
            # Save final result
            new_img = nib.Nifti1Image(result, img.affine)
            output_path = output_folder / f"{file_path.stem}_kmeans_split.nii.gz"
            nib.save(new_img, str(output_path))
            print(f"Saved split segmentation to {output_path}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}:")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return True

def process_segmentations(folder_path, output_folder=None, file_suffix="_bin.nii.gz"):
    """Legacy function for backward compatibility - processes all files in a folder"""
    folder_path = Path(folder_path)
    
    if output_folder is None:
        # Default output folder is a subdirectory of the base directory
        output_folder = folder_path.parent / "Splitted_annotation_masks"
    else:
        output_folder = Path(output_folder)
        
    output_folder.mkdir(exist_ok=True)
    
    vis_folder = output_folder / "visualizations"
    vis_folder.mkdir(exist_ok=True)
    
    steps_folder = output_folder / "step_visualizations"
    steps_folder.mkdir(exist_ok=True)
    
    nifti_files = [f for f in folder_path.glob(f'*{file_suffix}') if '_split' not in f.name]
    
    if not nifti_files:
        print(f"No files with suffix '{file_suffix}' found in {folder_path}")
        return False
    
    print(f"Found {len(nifti_files)} files to process")
    
    for nifti_file in nifti_files:
        print(f"\nProcessing {nifti_file.name}...")
        
        try:
            img = nib.load(str(nifti_file))
            data = img.get_fdata()
            
            # Find the best slice by examining all three orientations
            best_axis, best_slice_idx = find_best_slice(data)
            
            if best_axis is None:
                print(f"Skipping {nifti_file.name} - no content found")
                continue
                
            # Extract the best 2D slice based on the identified orientation
            if best_axis == 0:
                # Original orientation (axial)
                mask = data[best_slice_idx].astype(bool)
                print(f"Using axial slice {best_slice_idx} (max content)")
            elif best_axis == 1:
                # Sagittal orientation
                mask = data[:, best_slice_idx].astype(bool)
                print(f"Using sagittal slice {best_slice_idx} (max content)")
            else:
                # Coronal orientation
                mask = data[:, :, best_slice_idx].astype(bool)
                print(f"Using coronal slice {best_slice_idx} (max content)")
            
            # Process each connected component
            labeled_components, n_components = ndimage.label(mask)
            result = np.zeros_like(data)
            current_label = 1
            
            print(f"Found {n_components} connected components")
            
            for comp_idx in range(1, n_components + 1):
                print(f"Processing component {comp_idx}/{n_components}")
                component = labeled_components == comp_idx
                component_size = np.count_nonzero(component)
                print(f"Component size: {component_size} pixels")
                
                processed_comp, potential_seeds, seed_points, initial_regions, final_regions = process_component(
                    component, 
                    current_label
                )
                
                # Create visualization for every component
                steps_output_path = steps_folder / f"{nifti_file.stem}_component{comp_idx}_steps.png"
                visualize_steps(
                    component,
                    potential_seeds,
                    seed_points,
                    initial_regions,
                    final_regions,
                    f"Processing Steps for Component {comp_idx}",
                    steps_output_path
                )
                
                # Place the processed component back in the right orientation
                if best_axis == 0:
                    # Original orientation (axial)
                    result[best_slice_idx][component] = processed_comp[component]
                elif best_axis == 1:
                    # Sagittal orientation
                    result[:, best_slice_idx][component] = processed_comp[component]
                else:
                    # Coronal orientation
                    result[:, :, best_slice_idx][component] = processed_comp[component]
                
                current_label = result.max() + 1
            
            # Save final result
            new_img = nib.Nifti1Image(result, img.affine)
            output_path = output_folder / f"{nifti_file.stem}_kmeans_split.nii.gz"
            nib.save(new_img, str(output_path))
            print(f"Saved split segmentation to {output_path}")
            
        except Exception as e:
            print(f"Error processing {nifti_file.name}:")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split segmentation masks into smaller regions")
    parser.add_argument("--input_folder", type=str, default=None, 
                        help="Folder containing segmentation masks to split (legacy mode)")
    parser.add_argument("--output_folder", type=str, default=None, 
                        help="Folder to save split masks and visualizations (legacy mode)")
    parser.add_argument("--suffix", type=str, default="_bin.nii.gz",
                        help="Suffix of files to process (legacy mode, default: _bin.nii.gz)")
    parser.add_argument("--gui", action="store_true",
                        help="Use GUI file selection (default behavior)")
    
    args = parser.parse_args()
    
    # Default to GUI mode unless legacy folder arguments are provided
    use_gui = args.gui or (args.input_folder is None)
    
    if use_gui:
        print("Opening file selection dialog...")
        selected_files = select_input_files()
        
        if selected_files:
            print(f"Selected {len(selected_files)} files:")
            for file_path in selected_files:
                print(f"  - {Path(file_path).name}")
            
            success = process_selected_files(selected_files)
            if success:
                print("\nProcessing completed successfully!")
                print(f"Results saved to: {Path.cwd() / 'Splitted_annotation_masks'}")
        else:
            print("No files selected. Exiting.")
    
    else:
        # Legacy folder-based mode
        if args.input_folder is None:
            # Use script's directory as base directory
            base_dir = Path(__file__).parent
            input_folder = base_dir / "Transformation_results"
        else:
            input_folder = Path(args.input_folder)
        
        # Set default output folder if not provided
        if args.output_folder is None:
            # Use script's directory as base directory
            base_dir = Path(__file__).parent
            output_folder = base_dir / "Splitted_annotation_masks"
        else:
            output_folder = Path(args.output_folder)
            
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"File suffix: {args.suffix}")
        
        process_segmentations(input_folder, output_folder, args.suffix)
