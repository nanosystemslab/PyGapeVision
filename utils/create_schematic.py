import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

def create_schematic():
    """Create a schematic diagram showing the measurement points on the hook."""

    # Load the first frame
    img = cv2.imread('first_frame.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Original image with annotations
    ax1.imshow(img_rgb)
    ax1.set_title('Hook Under Tension - Original Frame', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Approximate coordinates for key points (based on visual inspection)
    # These will be refined by the tracking algorithm
    shaft_point = (320, 430)  # Green section on shaft (reference point)
    tip_point = (720, 580)     # Green section at tip

    # Draw circles at key points
    circle1 = Circle(shaft_point, 20, color='red', fill=False, linewidth=3, label='Reference Point (Shaft)')
    circle2 = Circle(tip_point, 20, color='blue', fill=False, linewidth=3, label='Tracking Point (Tip)')
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)

    # Draw the measurement line
    arrow = FancyArrowPatch(shaft_point, tip_point,
                           arrowstyle='<->', mutation_scale=30,
                           linewidth=2, color='yellow',
                           linestyle='--', label='Gape Distance (d)')
    ax1.add_patch(arrow)

    # Add text annotations
    mid_point = ((shaft_point[0] + tip_point[0])/2, (shaft_point[1] + tip_point[1])/2)
    ax1.text(mid_point[0], mid_point[1]-30, 'Gape Distance (d)',
            fontsize=12, color='yellow', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax1.text(shaft_point[0], shaft_point[1]-40, 'Reference\nPoint',
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            ha='center')

    ax1.text(tip_point[0], tip_point[1]+50, 'Tip\nPoint',
            fontsize=10, color='blue', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            ha='center')

    # Right plot: Schematic diagram
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.set_title('Measurement Schematic', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Draw simplified hook shape
    # Hook shaft (vertical)
    ax2.plot([2, 2], [7, 3.5], 'k-', linewidth=4)
    # Hook bend
    theta = np.linspace(np.pi, 0, 50)
    r = 1.5
    x_bend = 3.5 + r * np.cos(theta)
    y_bend = 3.5 + r * np.sin(theta)
    ax2.plot(x_bend, y_bend, 'k-', linewidth=4)
    # Hook point
    ax2.plot([5, 6], [3.5, 2], 'k-', linewidth=4)

    # Mark green painted sections
    ax2.plot([2, 2], [4.5, 3.8], 'g-', linewidth=8, label='Green Paint (Shaft)')
    ax2.plot([5.5, 6], [2.75, 2], 'g-', linewidth=8, label='Green Paint (Tip)')

    # Mark measurement points
    ref_point = (2, 4.15)
    tip_point_schem = (5.75, 2.375)

    ax2.plot(ref_point[0], ref_point[1], 'ro', markersize=15,
            markeredgewidth=2, markeredgecolor='darkred', label='Reference Point')
    ax2.plot(tip_point_schem[0], tip_point_schem[1], 'bo', markersize=15,
            markeredgewidth=2, markeredgecolor='darkblue', label='Tip Point')

    # Draw measurement line with arrows
    ax2.annotate('', xy=tip_point_schem, xytext=ref_point,
                arrowprops=dict(arrowstyle='<->', lw=2, color='purple'))

    # Add distance label
    mid_x = (ref_point[0] + tip_point_schem[0]) / 2
    mid_y = (ref_point[1] + tip_point_schem[1]) / 2
    ax2.text(mid_x-0.5, mid_y, 'Gape\nDistance (d)', fontsize=12,
            fontweight='bold', color='purple',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Add labels
    ax2.text(2, 7.5, 'Hook Shaft', fontsize=11, ha='center', fontweight='bold')
    ax2.text(6, 1.5, 'Hook Tip', fontsize=11, ha='center', fontweight='bold')

    # Add legend
    ax2.legend(loc='upper right', fontsize=9)

    # Add description text box
    description = ("Measurement Approach:\n"
                  "1. Track green-painted shaft (reference point)\n"
                  "2. Track green-painted tip (moving point)\n"
                  "3. Calculate distance 'd' between points\n"
                  "4. Monitor 'd' vs. time as hook deforms")

    ax2.text(5, 8.5, description, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
            verticalalignment='top')

    plt.tight_layout()
    plt.savefig('measurement_schematic.png', dpi=150, bbox_inches='tight')
    print("Schematic diagram saved to measurement_schematic.png")
    plt.close()

if __name__ == "__main__":
    create_schematic()
