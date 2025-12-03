"""
Simple NFL Tracking Data Explorer
Run this first to understand your data structure
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def explore_tracking_data(csv_path):
    """
    Quick exploration of your tracking CSV
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*70)
    print("DATA STRUCTURE")
    print("="*70)
    
    # Basic info
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns:\n{df.columns.tolist()}")
    
    print("\n" + "="*70)
    print("SAMPLE DATA")
    print("="*70)
    print(df.head())
    
    print("\n" + "="*70)
    print("DATA STATISTICS")
    print("="*70)
    
    # Unique values
    print(f"\nUnique plays: {df['play_id'].nunique()}")
    print(f"Unique players: {df['nfl_player_id'].nunique()}")
    
    if 'team' in df.columns:
        print(f"Teams: {df['team'].unique()}")
    
    if 'position' in df.columns:
        print(f"\nPositions represented: {df['position'].unique()}")
    
    # Steps per play
    steps_per_play = df.groupby('play_id')['step'].nunique()
    print(f"\nSteps per play:")
    print(f"  Min: {steps_per_play.min()}")
    print(f"  Max: {steps_per_play.max()}")
    print(f"  Mean: {steps_per_play.mean():.1f}")
    
    # Position ranges
    print(f"\nPosition ranges:")
    print(f"  X: [{df['x_position'].min():.2f}, {df['x_position'].max():.2f}]")
    print(f"  Y: [{df['y_position'].min():.2f}, {df['y_position'].max():.2f}]")
    
    # Speed statistics
    print(f"\nSpeed (yards/second):")
    print(f"  Min: {df['speed'].min():.2f}")
    print(f"  Max: {df['speed'].max():.2f}")
    print(f"  Mean: {df['speed'].mean():.2f}")
    
    return df

def plot_single_frame(df, play_id=None, step=0):
    """
    Plot player positions for a single frame
    """
    if play_id is None:
        play_id = df['play_id'].iloc[0]
    
    # Get data for this frame
    frame_data = df[(df['play_id'] == play_id) & (df['step'] == step)]
    
    if len(frame_data) == 0:
        print(f"No data found for play {play_id}, step {step}")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Draw field
    field_rect = plt.Rectangle((0, 0), 120, 53.3, 
                               fill=True, facecolor='#196f0c', 
                               edgecolor='white', linewidth=2)
    ax.add_patch(field_rect)
    
    # Yard lines
    for x in range(10, 111, 10):
        ax.axvline(x, color='white', linewidth=1, alpha=0.5)
        ax.text(x, -2, f"{x}", ha='center', color='white', fontsize=10)
    
    # Goal lines
    ax.axvline(10, color='white', linewidth=2)
    ax.axvline(110, color='white', linewidth=2)
    
    # Plot players
    for _, player in frame_data.iterrows():
        x = player['x_position']
        y = player['y_position']
        
        # Color by team if available
        if 'team' in player and pd.notna(player['team']):
            color = 'blue' if 'home' in str(player['team']).lower() else 'red'
        else:
            color = 'yellow'
        
        # Draw player
        circle = plt.Circle((x, y), 1.5, color=color, alpha=0.7, zorder=3)
        ax.add_patch(circle)
        
        # Jersey number
        if 'jersey_number' in player and pd.notna(player['jersey_number']):
            ax.text(x, y, str(int(player['jersey_number'])), 
                   ha='center', va='center', color='white', 
                   fontsize=10, fontweight='bold', zorder=4)
        
        # Speed arrow
        if 'direction' in player and pd.notna(player['direction']):
            direction = np.radians(player['direction'])
            speed = player['speed'] if 'speed' in player else 1
            dx = speed * 0.5 * np.cos(direction)
            dy = speed * 0.5 * np.sin(direction)
            ax.arrow(x, y, dx, dy, head_width=1, head_length=0.7,
                    fc=color, ec='white', linewidth=1.5, alpha=0.6, zorder=2)
    
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, 58)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (yards)', fontsize=12)
    ax.set_ylabel('Y Position (yards)', fontsize=12)
    ax.set_title(f'Play {play_id} - Step {step}\n({len(frame_data)} players)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax

def visualize_play_sequence(df, play_id=None, num_frames=9):
    """
    Show multiple frames from a play in a grid
    """
    if play_id is None:
        play_id = df['play_id'].iloc[0]
    
    play_data = df[df['play_id'] == play_id]
    steps = sorted(play_data['step'].unique())
    
    # Select evenly spaced steps
    step_indices = np.linspace(0, len(steps)-1, num_frames, dtype=int)
    selected_steps = [steps[i] for i in step_indices]
    
    # Create subplot grid
    rows = int(np.ceil(np.sqrt(num_frames)))
    cols = int(np.ceil(num_frames / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if num_frames > 1 else [axes]
    
    for idx, step in enumerate(selected_steps):
        frame_data = play_data[play_data['step'] == step]
        ax = axes[idx]
        
        # Draw field
        field_rect = plt.Rectangle((0, 0), 120, 53.3, 
                                   fill=True, facecolor='#196f0c',
                                   edgecolor='white', linewidth=1)
        ax.add_patch(field_rect)
        
        # Yard lines
        for x in range(10, 111, 10):
            ax.axvline(x, color='white', linewidth=0.5, alpha=0.5)
        
        # Players
        for _, player in frame_data.iterrows():
            x, y = player['x_position'], player['y_position']
            color = 'blue' if 'home' in str(player.get('team', '')).lower() else 'red'
            
            circle = plt.Circle((x, y), 1.5, color=color, alpha=0.7)
            ax.add_patch(circle)
            
            if 'jersey_number' in player and pd.notna(player['jersey_number']):
                ax.text(x, y, str(int(player['jersey_number'])),
                       ha='center', va='center', color='white',
                       fontsize=6, fontweight='bold')
        
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        ax.set_aspect('equal')
        ax.set_title(f'Step {step}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    for idx in range(num_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Play {play_id} Sequence', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_player_trajectories(df, play_id=None):
    """
    Plot trajectories of all players in a play
    """
    if play_id is None:
        play_id = df['play_id'].iloc[0]
    
    play_data = df[df['play_id'] == play_id]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Draw field
    field_rect = plt.Rectangle((0, 0), 120, 53.3,
                               fill=True, facecolor='#196f0c',
                               edgecolor='white', linewidth=2)
    ax.add_patch(field_rect)
    
    # Yard lines
    for x in range(10, 111, 10):
        ax.axvline(x, color='white', linewidth=1, alpha=0.3)
    
    # Plot trajectory for each player
    for player_id in play_data['nfl_player_id'].unique():
        player_traj = play_data[play_data['nfl_player_id'] == player_id].sort_values('step')
        
        x = player_traj['x_position'].values
        y = player_traj['y_position'].values
        
        # Color by team
        team = player_traj['team'].iloc[0] if 'team' in player_traj.columns else 'unknown'
        color = 'blue' if 'home' in str(team).lower() else 'red'
        
        # Plot trajectory
        ax.plot(x, y, '-', color=color, alpha=0.6, linewidth=2)
        
        # Mark start and end
        ax.plot(x[0], y[0], 'o', color=color, markersize=8, alpha=0.8)
        ax.plot(x[-1], y[-1], 's', color=color, markersize=8, alpha=0.8)
        
        # Jersey number at end position
        if 'jersey_number' in player_traj.columns:
            jersey = player_traj['jersey_number'].iloc[0]
            ax.text(x[-1], y[-1], str(int(jersey)),
                   ha='center', va='center', color='white',
                   fontsize=8, fontweight='bold')
    
    ax.set_xlim(-5, 125)
    ax.set_ylim(-5, 58)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position (yards)', fontsize=12)
    ax.set_ylabel('Y Position (yards)', fontsize=12)
    ax.set_title(f'Player Trajectories - Play {play_id}', fontsize=14, fontweight='bold')
    ax.legend(['Start', 'End'], loc='upper right')
    
    plt.tight_layout()
    return fig


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    print("\n" + "="*70)
    print("NFL TRACKING DATA QUICK EXPLORER")
    print("="*70)
    
    # Path to your CSV
    csv_path = "nfl-player-contact-detection/train_player_tracking.csv"
    
    # Load and explore
    df = explore_tracking_data(csv_path)
    
    # Get a sample play to visualize
    print("\n" + "="*70)
    print("VISUALIZATION OPTIONS")
    print("="*70)
    
    sample_play = df['play_id'].iloc[0]
    print(f"\nUsing sample play: {sample_play}")
    
    choice = input("\nWhat would you like to see?\n"
                   "1. Single frame snapshot\n"
                   "2. Play sequence (grid of frames)\n"
                   "3. Player trajectories\n"
                   "4. All of the above\n"
                   "Choice (1-4): ").strip()
    
    # Create output directory
    output_dir = Path('./nfl_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    if choice in ['1', '4']:
        print("\nGenerating single frame view...")
        fig, ax = plot_single_frame(df, sample_play, step=0)
        output_path = output_dir / 'single_frame.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    
    if choice in ['2', '4']:
        print("\nGenerating play sequence...")
        fig = visualize_play_sequence(df, sample_play, num_frames=9)
        output_path = output_dir / 'play_sequence.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    
    if choice in ['3', '4']:
        print("\nGenerating player trajectories...")
        fig = plot_player_trajectories(df, sample_play)
        output_path = output_dir / 'player_trajectories.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
        plt.close()
    
    print("\n" + "="*70)
    print("DONE! Check the ./nfl_visualizations/ folder for outputs.")
    print("="*70)
    print("\nGenerated files:")
    if choice in ['1', '4']:
        print("  - single_frame.png")
    if choice in ['2', '4']:
        print("  - play_sequence.png")
    if choice in ['3', '4']:
        print("  - player_trajectories.png")
    print("\nNext steps:")
    print("1. Use nfl_tracking_visualizer.py for interactive exploration")
    print("2. Check coordinate system alignment with your videos")
    print("3. Begin camera calibration using field markings")
    print("="*70)