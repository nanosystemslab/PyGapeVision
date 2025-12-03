"""
Point tracker for green-painted markers on hook tension tests.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import json


class GreenPointTracker:
    """Track green-painted points on a hook during tension testing."""

    def __init__(self, hsv_lower=(35, 100, 50), hsv_upper=(55, 255, 255),
                 use_simple_tracking=False, exclude_right_pixels=0):
        """
        Initialize the tracker.

        Args:
            hsv_lower: Lower bound for green color in HSV space (H, S, V)
            hsv_upper: Upper bound for green color in HSV space (H, S, V)
            use_simple_tracking: If True, use simple "two largest contours" method
                                instead of spatial zone detection
            exclude_right_pixels: Number of pixels to exclude from the right edge
                                 (useful for excluding rulers, overlays, etc.)

        Default range targets vibrant yellowish-green markers (Overgrowth/Poisonous Potion).
        Excludes:
        - Bluer greens (Kryptonite Green at H≈65)
        - Gray/desaturated colors (min S=100)
        """
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        self.use_simple_tracking = use_simple_tracking
        self.exclude_right_pixels = exclude_right_pixels

    def find_green_regions(self, frame: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Find green regions in the frame.

        Args:
            frame: Input BGR frame

        Returns:
            mask: Binary mask of green regions
            contours_info: List of contour information (area, centroid, bounding box)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for green color
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Exclude right edge if specified (for rulers, overlays, etc.)
        if self.exclude_right_pixels > 0:
            height, width = mask.shape
            mask[:, width - self.exclude_right_pixels:] = 0

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract contour information
        contours_info = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Filter small noise
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x, y, w, h = cv2.boundingRect(cnt)

                    contours_info.append({
                        'centroid': (cx, cy),
                        'area': area,
                        'bbox': (x, y, w, h),
                        'contour': cnt
                    })

        # Sort by area (largest first)
        contours_info.sort(key=lambda x: x['area'], reverse=True)

        return mask, contours_info

    def validate_tip_edges(self, contour_info: dict, frame: np.ndarray = None,
                           min_edge_separation: int = 15) -> bool:
        """
        Validate that a tip candidate has two distinct edges at the boundary
        between green paint and metal/background (left and right sides of hook point).

        Args:
            contour_info: Contour information dictionary
            frame: Original frame for edge detection (optional, for enhanced validation)
            min_edge_separation: Minimum pixel distance between left and right edges

        Returns:
            True if two distinct boundary edges are detected, False otherwise
        """
        contour = contour_info['contour']

        # Get bounding box and extreme points
        x, y, w, h = contour_info['bbox']
        leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
        rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

        # Calculate horizontal span (left to right boundary edge)
        horizontal_span = rightmost[0] - leftmost[0]
        vertical_span = bottommost[1] - topmost[1]

        # Basic geometry checks
        if horizontal_span < min_edge_separation:
            return False

        # Check aspect ratio - tip should have some width
        if vertical_span > 0:
            aspect_ratio = horizontal_span / vertical_span
            if aspect_ratio < 0.3:  # Too narrow/vertical
                return False

        # Enhanced validation: Check for boundary edges if frame is provided
        if frame is not None:
            # Extract ROI with padding
            padding = 15
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)

            roi = frame[y1:y2, x1:x2]
            roi_height, roi_width = roi.shape[:2]

            # Skip if ROI is too small
            if roi_width < 20 or roi_height < 10:
                return True  # Accept small regions without edge check

            # Convert ROI to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Apply Canny edge detection with more sensitive parameters
            edges = cv2.Canny(gray_roi, 30, 100)

            # Look for edges on left and right portions of the region
            # For hook tip, we expect edges at the boundaries where green meets metal
            left_portion_width = max(5, roi_width // 4)
            right_portion_width = max(5, roi_width // 4)

            # Check left portion for left edge
            left_portion = edges[:, :left_portion_width]
            left_edge_strength = np.sum(left_portion > 0)

            # Check right portion for right edge
            right_portion = edges[:, -right_portion_width:]
            right_edge_strength = np.sum(right_portion > 0)

            # Scale minimum edge pixels based on region size
            # Smaller regions need fewer edge pixels
            min_edge_pixels = max(3, int(roi_height * 0.15))

            # Both edges should have some edge pixels at the boundaries
            if left_edge_strength < min_edge_pixels or right_edge_strength < min_edge_pixels:
                return False

        return True

    def identify_shaft_and_tip(self, contours_info: List[dict],
                              frame_width: int = 1920,
                              frame_height: int = 1080,
                              prev_shaft: Optional[Tuple[int, int]] = None,
                              prev_tip: Optional[Tuple[int, int]] = None,
                              search_radius: int = 150,
                              frame: Optional[np.ndarray] = None) -> Tuple[Optional[dict], Optional[dict]]:
        """
        Identify which green region is the shaft and which is the tip.

        Uses temporal tracking: if previous positions are provided, searches near
        those locations first to maintain tracking consistency.

        Args:
            contours_info: List of contour information
            frame_width: Width of the frame
            frame_height: Height of the frame
            prev_shaft: Previous shaft centroid (x, y) for temporal tracking
            prev_tip: Previous tip centroid (x, y) for temporal tracking
            search_radius: Pixel radius to search around previous position
            frame: Original frame for enhanced edge validation (optional)

        Returns:
            shaft_info: Information about the shaft marker
            tip_info: Information about the tip marker
        """
        if len(contours_info) < 2:
            return None, None

        # Simple tracking mode: just use two largest contours
        if self.use_simple_tracking:
            # Already sorted by area (largest first)
            candidates = contours_info[:2]

            # Determine which is shaft (upper) and which is tip (lower) based on y-coordinate
            if candidates[0]['centroid'][1] < candidates[1]['centroid'][1]:
                shaft_info = candidates[0]
                tip_info = candidates[1]
            else:
                shaft_info = candidates[1]
                tip_info = candidates[0]

            return shaft_info, tip_info

        # Filter candidates based on position and area
        left_boundary = frame_width * 0.65
        min_area = 200
        max_area = 15000

        valid_candidates = [
            c for c in contours_info
            if c['centroid'][0] < left_boundary
            and min_area < c['area'] < max_area
        ]

        if len(valid_candidates) < 2:
            return None, None

        # Sort by area
        valid_candidates.sort(key=lambda x: x['area'], reverse=True)

        # TEMPORAL TRACKING: If we have previous positions, use proximity-based search
        if prev_shaft is not None and prev_tip is not None:
            shaft_info = None
            tip_info = None

            # Find closest region to previous shaft position
            shaft_candidates_near = []
            for c in valid_candidates[:20]:
                cx, cy = c['centroid']
                dist = ((cx - prev_shaft[0])**2 + (cy - prev_shaft[1])**2)**0.5
                if dist < search_radius:
                    shaft_candidates_near.append((dist, c))

            # Find closest region to previous tip position
            tip_candidates_near = []
            for c in valid_candidates[:20]:
                cx, cy = c['centroid']
                dist = ((cx - prev_tip[0])**2 + (cy - prev_tip[1])**2)**0.5
                if dist < search_radius:
                    tip_candidates_near.append((dist, c))

            # Select closest matches
            if shaft_candidates_near:
                shaft_candidates_near.sort(key=lambda x: x[0])  # Sort by distance
                shaft_info = shaft_candidates_near[0][1]

            if tip_candidates_near:
                tip_candidates_near.sort(key=lambda x: x[0])  # Sort by distance
                # Validate tip has two distinct edges before accepting
                candidate_tip = tip_candidates_near[0][1]
                if self.validate_tip_edges(candidate_tip, frame=frame):
                    tip_info = candidate_tip

            # If both found via temporal tracking, return them
            if shaft_info and tip_info:
                return shaft_info, tip_info

        # INITIAL DETECTION or RECOVERY: Use spatial zones
        shaft_candidates = []
        tip_candidates = []

        for c in valid_candidates[:15]:
            cx, cy = c['centroid']

            # Shaft region: upper-middle area (initial: ~724, 460)
            if 380 < cy < 520 and 620 < cx < 850:
                shaft_candidates.append(c)

            # Tip region: lower-right (initial: ~902, 707)
            if 650 < cy < 800 and 800 < cx < 1000:
                # Validate tip has two distinct edges at boundary
                if self.validate_tip_edges(c, frame=frame):
                    tip_candidates.append(c)

        # Select largest from each category
        shaft_info = shaft_candidates[0] if shaft_candidates else None
        tip_info = tip_candidates[0] if tip_candidates else None

        # Fallback
        if not shaft_info or not tip_info:
            if len(valid_candidates) >= 2:
                candidates = valid_candidates[:2]
                # Determine which is shaft (upper) and which is tip (lower)
                if candidates[0]['centroid'][1] < candidates[1]['centroid'][1]:
                    shaft_info = candidates[0]
                    # Validate tip has two distinct edges at boundary
                    if self.validate_tip_edges(candidates[1], frame=frame):
                        tip_info = candidates[1]
                else:
                    shaft_info = candidates[1]
                    # Validate tip has two distinct edges at boundary
                    if self.validate_tip_edges(candidates[0], frame=frame):
                        tip_info = candidates[0]

        return shaft_info, tip_info

    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            point1: (x, y) coordinates of first point
            point2: (x, y) coordinates of second point

        Returns:
            distance: Distance in pixels
        """
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def annotate_frame(self, frame: np.ndarray, shaft_info: dict, tip_info: dict,
                       distance: float, pixels_per_mm: float = None) -> np.ndarray:
        """
        Draw annotations on the frame.

        Args:
            frame: Input frame
            shaft_info: Information about shaft marker
            tip_info: Information about tip marker
            distance: Calculated distance in pixels
            pixels_per_mm: Optional calibration factor to display in mm

        Returns:
            annotated_frame: Frame with annotations
        """
        annotated = frame.copy()

        # Draw shaft marker (red circle)
        if shaft_info:
            cv2.circle(annotated, shaft_info['centroid'], 10, (0, 0, 255), 2)
            cv2.putText(annotated, "Shaft", (shaft_info['centroid'][0] - 30, shaft_info['centroid'][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw tip marker (blue circle)
        if tip_info:
            cv2.circle(annotated, tip_info['centroid'], 10, (255, 0, 0), 2)
            cv2.putText(annotated, "Tip", (tip_info['centroid'][0] - 20, tip_info['centroid'][1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw distance line
        if shaft_info and tip_info:
            cv2.line(annotated, shaft_info['centroid'], tip_info['centroid'], (0, 255, 255), 2)

            # Draw distance text (in mm if calibrated, otherwise pixels)
            mid_x = (shaft_info['centroid'][0] + tip_info['centroid'][0]) // 2
            mid_y = (shaft_info['centroid'][1] + tip_info['centroid'][1]) // 2

            if pixels_per_mm:
                distance_mm = distance / pixels_per_mm
                cv2.putText(annotated, f"d = {distance_mm:.1f} mm",
                           (mid_x - 60, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(annotated, f"d = {distance:.1f} px",
                           (mid_x - 50, mid_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return annotated

    def overlay_force_displacement_curve(self, frame: np.ndarray,
                                         shimadzu_df,
                                         current_time: float,
                                         time_offset: float) -> np.ndarray:
        """
        Overlay force-displacement curve in upper right corner of frame.

        Args:
            frame: Input frame
            shimadzu_df: DataFrame with Force and Stroke data
            current_time: Current video time in seconds
            time_offset: Time offset to sync with mechanical data

        Returns:
            frame: Frame with overlay
        """
        if shimadzu_df is None or len(shimadzu_df) == 0:
            return frame

        # Define overlay position and size (upper right corner) - increased size for better visibility
        overlay_width = 450
        overlay_height = 350
        margin = 20
        x_start = frame.shape[1] - overlay_width - margin
        y_start = margin

        # Create white background for overlay
        cv2.rectangle(frame, (x_start, y_start),
                     (x_start + overlay_width, y_start + overlay_height),
                     (255, 255, 255), -1)
        cv2.rectangle(frame, (x_start, y_start),
                     (x_start + overlay_width, y_start + overlay_height),
                     (0, 0, 0), 2)

        # Get data up to current time
        mech_time = current_time + time_offset
        current_data = shimadzu_df[shimadzu_df['Time'] <= mech_time]

        if len(current_data) == 0:
            return frame

        stroke = current_data['Stroke'].values
        force = current_data['Force'].values

        # Normalize to plot coordinates - more space at top for text
        plot_margin_top = 70
        plot_margin_other = 40
        plot_width = overlay_width - 2 * plot_margin_other
        plot_height = overlay_height - plot_margin_top - plot_margin_other

        stroke_min, stroke_max = shimadzu_df['Stroke'].min(), shimadzu_df['Stroke'].max()
        force_min, force_max = shimadzu_df['Force'].min(), shimadzu_df['Force'].max()

        if stroke_max > stroke_min and force_max > force_min:
            # Normalize coordinates
            x_norm = (stroke - stroke_min) / (stroke_max - stroke_min)
            y_norm = (force - force_min) / (force_max - force_min)

            x_plot = (x_norm * plot_width + x_start + plot_margin_other).astype(int)
            y_plot = (y_start + overlay_height - plot_margin_other - y_norm * plot_height).astype(int)

            # Draw curve
            for i in range(1, len(x_plot)):
                cv2.line(frame, (x_plot[i-1], y_plot[i-1]),
                        (x_plot[i], y_plot[i]), (0, 0, 255), 2)

            # Draw current point
            if len(x_plot) > 0:
                cv2.circle(frame, (x_plot[-1], y_plot[-1]), 6, (255, 0, 0), -1)

            # Add title in upper section (larger font)
            cv2.putText(frame, "Force-Displacement",
                       (x_start + 15, y_start + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

            # Add current values in upper section (larger font, moved above plot)
            if len(current_data) > 0:
                current_force = force[-1]
                current_stroke = stroke[-1]
                cv2.putText(frame, f"F: {current_force:.1f} N",
                           (x_start + 15, y_start + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                cv2.putText(frame, f"S: {current_stroke:.2f} mm",
                           (x_start + 230, y_start + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

        return frame


class VideoAnalyzer:
    """Analyze video to track gape distance over time."""

    def __init__(self, video_path: str, tracker: GreenPointTracker):
        """
        Initialize the video analyzer.

        Args:
            video_path: Path to video file
            tracker: GreenPointTracker instance
        """
        self.video_path = video_path
        self.tracker = tracker
        self.results = {
            'frame_number': [],
            'time_seconds': [],
            'shaft_x': [],
            'shaft_y': [],
            'tip_x': [],
            'tip_y': [],
            'distance_pixels': []
        }

    def process_video(self, output_video_path: Optional[str] = None,
                     frame_skip: int = 1,
                     fps_override: Optional[float] = None,
                     rotate_90_cw: bool = False,
                     shimadzu_df = None,
                     time_offset: float = 0.0,
                     pixels_per_mm: Optional[float] = None,
                     init_shaft_pos: Optional[Tuple[int, int]] = None,
                     init_tip_pos: Optional[Tuple[int, int]] = None) -> dict:
        """
        Process the entire video and track points.

        Args:
            output_video_path: Path to save annotated video (optional)
            frame_skip: Process every Nth frame (1 = process all frames)
            fps_override: Override frame rate (fps). If None, uses video metadata
            rotate_90_cw: Rotate output video 90 degrees clockwise
            shimadzu_df: DataFrame with force/displacement data for overlay (optional)
            time_offset: Time offset for synchronizing with shimadzu data
            pixels_per_mm: Calibration factor to display measurements in mm
            init_shaft_pos: Initial shaft position (x, y) for manual tracking initialization
            init_tip_pos: Initial tip position (x, y) for manual tracking initialization

        Returns:
            results: Dictionary containing tracking data
        """
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        # Get video properties
        fps = fps_override if fps_override is not None else cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video properties: {width}x{height} @ {fps} fps, {total_frames} frames")

        # Adjust dimensions for rotation
        output_width = height if rotate_90_cw else width
        output_height = width if rotate_90_cw else height

        # Setup video writer if output path is provided
        out = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps/frame_skip, (output_width, output_height))

        frame_number = 0
        processed_frames = 0

        # Track previous positions for temporal tracking
        # Use manual initialization if provided
        prev_shaft_pos = init_shaft_pos
        prev_tip_pos = init_tip_pos

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if needed
            if frame_number % frame_skip != 0:
                frame_number += 1
                continue

            # Find green regions
            mask, contours_info = self.tracker.find_green_regions(frame)

            # Identify shaft and tip using temporal tracking
            shaft_info, tip_info = self.tracker.identify_shaft_and_tip(
                contours_info,
                frame_width=width,
                frame_height=height,
                prev_shaft=prev_shaft_pos,
                prev_tip=prev_tip_pos,
                frame=frame
            )

            # Fallback: If manual positions provided but detection failed, search for ANY green near manual positions
            if (shaft_info is None or tip_info is None) and (init_shaft_pos is not None and init_tip_pos is not None):
                # Try to find the closest green region to each manual position
                large_search_radius = 300  # Much larger search radius for manual mode

                if shaft_info is None and prev_shaft_pos is not None and len(contours_info) > 0:
                    # Find closest contour to shaft position
                    closest_shaft = None
                    min_dist = float('inf')
                    for c in contours_info:
                        cx, cy = c['centroid']
                        dist = ((cx - prev_shaft_pos[0])**2 + (cy - prev_shaft_pos[1])**2)**0.5
                        if dist < min_dist and dist < large_search_radius:
                            min_dist = dist
                            closest_shaft = c
                    if closest_shaft:
                        shaft_info = closest_shaft

                if tip_info is None and prev_tip_pos is not None and len(contours_info) > 0:
                    # Find closest contour to tip position
                    closest_tip = None
                    min_dist = float('inf')
                    for c in contours_info:
                        cx, cy = c['centroid']
                        # Make sure it's not the same as shaft
                        if shaft_info and c == shaft_info:
                            continue
                        dist = ((cx - prev_tip_pos[0])**2 + (cy - prev_tip_pos[1])**2)**0.5
                        if dist < min_dist and dist < large_search_radius:
                            min_dist = dist
                            closest_tip = c
                    if closest_tip:
                        tip_info = closest_tip

            # Calculate distance
            distance = 0.0
            if shaft_info and tip_info:
                distance = self.tracker.calculate_distance(
                    shaft_info['centroid'],
                    tip_info['centroid']
                )

                # Store results
                self.results['frame_number'].append(frame_number)
                self.results['time_seconds'].append(frame_number / fps)
                self.results['shaft_x'].append(shaft_info['centroid'][0])
                self.results['shaft_y'].append(shaft_info['centroid'][1])
                self.results['tip_x'].append(tip_info['centroid'][0])
                self.results['tip_y'].append(tip_info['centroid'][1])
                self.results['distance_pixels'].append(distance)

                # Update previous positions for next frame
                prev_shaft_pos = shaft_info['centroid']
                prev_tip_pos = tip_info['centroid']

            # Rotate FIRST if requested (before adding any overlays/text)
            frame_to_annotate = frame
            if rotate_90_cw and out:
                frame_to_annotate = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # Also need to rotate the tracking point coordinates
                if shaft_info:
                    # Transform coordinates for rotated frame: (x, y) -> (height - y, x)
                    orig_x, orig_y = shaft_info['centroid']
                    shaft_info = shaft_info.copy()
                    shaft_info['centroid'] = (height - orig_y, orig_x)
                if tip_info:
                    orig_x, orig_y = tip_info['centroid']
                    tip_info = tip_info.copy()
                    tip_info['centroid'] = (height - orig_y, orig_x)

            # Annotate frame (after rotation)
            annotated_frame = self.tracker.annotate_frame(frame_to_annotate, shaft_info, tip_info, distance, pixels_per_mm)

            # Add force-displacement overlay if mechanical data provided (after rotation)
            if shimadzu_df is not None:
                current_time = frame_number / fps
                annotated_frame = self.tracker.overlay_force_displacement_curve(
                    annotated_frame, shimadzu_df, current_time, time_offset
                )

            # Write to output video
            if out:
                out.write(annotated_frame)

            processed_frames += 1
            if processed_frames % 100 == 0:
                print(f"Processed {processed_frames} frames ({frame_number}/{total_frames})")

            frame_number += 1

        cap.release()
        if out:
            out.release()

        frames_tracked = len(self.results['frame_number'])
        print(f"Processing complete. Tracked {frames_tracked} frames.")

        # Add tracking method metadata
        if self.tracker.use_simple_tracking:
            self.results['tracking_method'] = 'simple'
        elif init_shaft_pos is not None and init_tip_pos is not None:
            self.results['tracking_method'] = 'manual_init'
        else:
            self.results['tracking_method'] = 'spatial_zones'

        self.results['frames_tracked'] = frames_tracked

        return self.results

    def save_results(self, output_json_path: str):
        """Save results to JSON file."""
        with open(output_json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_json_path}")

    def get_results_array(self) -> np.ndarray:
        """Get results as numpy array for further analysis."""
        return np.array([
            self.results['time_seconds'],
            self.results['distance_pixels'],
            self.results['shaft_x'],
            self.results['shaft_y'],
            self.results['tip_x'],
            self.results['tip_y']
        ]).T
