"""
Calibration data management for PyGapeVision.

Loads and manages calibration values (pixels per mm) from JSON file.
"""

import json
from pathlib import Path
from typing import Optional, Dict


class CalibrationManager:
    """Manage calibration data for video analysis."""
    
    def __init__(self, calibration_file: str = "results/calibration/calibration.json"):
        """
        Initialize calibration manager.
        
        Args:
            calibration_file: Path to calibration JSON file
        """
        self.calibration_file = Path(calibration_file)
        self.data = self._load_calibration()
    
    def _load_calibration(self) -> Dict:
        """Load calibration data from JSON file."""
        if not self.calibration_file.exists():
            print(f"Warning: Calibration file not found: {self.calibration_file}")
            return self._default_calibration()
        
        try:
            with open(self.calibration_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading calibration file: {e}")
            return self._default_calibration()
    
    def _default_calibration(self) -> Dict:
        """Return default calibration structure."""
        return {
            "version": "1.0",
            "default": {
                "pixels_per_mm": 12.103,
                "unit": "pixels/mm"
            },
            "batch_specific": {},
            "video_specific": {}
        }
    
    def get_pixels_per_mm(self, video_name: Optional[str] = None, 
                          batch_name: Optional[str] = None) -> float:
        """
        Get pixels per mm calibration value.
        
        Priority order:
        1. Video-specific calibration
        2. Batch-specific calibration
        3. Default calibration
        
        Args:
            video_name: Video filename (e.g., "A1.mp4")
            batch_name: Batch name (e.g., "Batch_A")
        
        Returns:
            Calibration value in pixels per mm
        """
        # Check video-specific
        if video_name and video_name in self.data.get("video_specific", {}):
            return self.data["video_specific"][video_name]["pixels_per_mm"]
        
        # Check batch-specific
        if batch_name and batch_name in self.data.get("batch_specific", {}):
            return self.data["batch_specific"][batch_name]["pixels_per_mm"]
        
        # Return default
        return self.data.get("default", {}).get("pixels_per_mm", 12.103)
    
    def set_calibration(self, pixels_per_mm: float, 
                       video_name: Optional[str] = None,
                       batch_name: Optional[str] = None,
                       notes: str = "") -> None:
        """
        Set calibration value.
        
        Args:
            pixels_per_mm: Calibration value
            video_name: If provided, sets video-specific calibration
            batch_name: If provided, sets batch-specific calibration
            notes: Optional notes about this calibration
        """
        if video_name:
            if "video_specific" not in self.data:
                self.data["video_specific"] = {}
            self.data["video_specific"][video_name] = {
                "pixels_per_mm": pixels_per_mm,
                "notes": notes
            }
        elif batch_name:
            if "batch_specific" not in self.data:
                self.data["batch_specific"] = {}
            self.data["batch_specific"][batch_name] = {
                "pixels_per_mm": pixels_per_mm,
                "notes": notes
            }
        else:
            # Set as default
            if "default" not in self.data:
                self.data["default"] = {}
            self.data["default"]["pixels_per_mm"] = pixels_per_mm
            self.data["default"]["notes"] = notes
        
        self.save()
    
    def save(self) -> None:
        """Save calibration data to JSON file."""
        # Ensure directory exists
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.calibration_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        
        print(f"Calibration saved to: {self.calibration_file}")


def load_calibration(video_name: Optional[str] = None,
                    batch_name: Optional[str] = None,
                    calibration_file: str = "results/calibration/calibration.json") -> float:
    """
    Convenience function to load calibration value.
    
    Args:
        video_name: Video filename
        batch_name: Batch name
        calibration_file: Path to calibration JSON
    
    Returns:
        Pixels per mm calibration value
    """
    manager = CalibrationManager(calibration_file)
    return manager.get_pixels_per_mm(video_name, batch_name)
