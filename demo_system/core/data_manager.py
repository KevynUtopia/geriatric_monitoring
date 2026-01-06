"""
Data manager - handles data storage and retrieval for detection results
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class DataManager:
    """Data manager for detection results and statistics"""
    
    def __init__(self):
        self.data_dir = "data"
        self.results_file = os.path.join(self.data_dir, "detection_results.json")
        self.stats_file = os.path.join(self.data_dir, "detection_stats.json")
        self.config_file = os.path.join(self.data_dir, "config.json")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data structures
        self.detection_results = []
        self.detection_stats = {}
        self.config = {}
        
        # Load existing data
        self.load_data()
    
    def load_data(self):
        """Load existing data from files"""
        try:
            # Load detection results
            if os.path.exists(self.results_file):
                with open(self.results_file, 'r', encoding='utf-8') as f:
                    self.detection_results = json.load(f)
            else:
                self.detection_results = []
            
            # Load detection statistics
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r', encoding='utf-8') as f:
                    self.detection_stats = json.load(f)
            else:
                self.detection_stats = self._init_default_stats()
            
            # Load configuration
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = self._init_default_config()
            
            print("Data loaded successfully")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.detection_results = []
            self.detection_stats = self._init_default_stats()
            self.config = self._init_default_config()
    
    def save_data(self):
        """Save data to files"""
        try:
            # Save detection results
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(self.detection_results, f, indent=2, ensure_ascii=False)
            
            # Save detection statistics
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.detection_stats, f, indent=2, ensure_ascii=False)
            
            # Save configuration
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            print("Data saved successfully")
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def _init_default_stats(self) -> Dict[str, Any]:
        """Initialize default statistics structure"""
        return {
            'total_detections': 0,
            'detection_types': {
                'pose_detection': 0,
                'object_detection': 0,
                'action_recognition': 0,
                'custom_detection': 0
            },
            'daily_stats': {},
            'weekly_stats': {},
            'monthly_stats': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def _init_default_config(self) -> Dict[str, Any]:
        """Initialize default configuration"""
        return {
            'detection_settings': {
                'confidence_threshold': 0.5,
                'max_detections_per_frame': 10,
                'save_detection_images': False
            },
            'ui_settings': {
                'language': 'en',
                'show_skeleton': True,
                'show_confidence_scores': True,
                'mirror_mode': True
            },
            'performance_settings': {
                'model_mode': 'balanced',
                'device': 'cpu',
                'max_fps': 30
            }
        }
    
    def add_detection_result(self, detection_type: str, results: Dict[str, Any], 
                           frame_info: Optional[Dict[str, Any]] = None):
        """Add a new detection result"""
        try:
            detection_record = {
                'timestamp': datetime.now().isoformat(),
                'detection_type': detection_type,
                'results': results,
                'frame_info': frame_info or {},
                'id': len(self.detection_results) + 1
            }
            
            self.detection_results.append(detection_record)
            
            # Update statistics
            self._update_stats(detection_type, results)
            
            # Keep only recent results (last 1000)
            if len(self.detection_results) > 1000:
                self.detection_results = self.detection_results[-1000:]
            
            # Auto-save periodically
            if len(self.detection_results) % 10 == 0:
                self.save_data()
            
        except Exception as e:
            print(f"Error adding detection result: {e}")
    
    def _update_stats(self, detection_type: str, results: Dict[str, Any]):
        """Update detection statistics"""
        try:
            # Update total detections
            self.detection_stats['total_detections'] += 1
            
            # Update detection type counts
            if detection_type in self.detection_stats['detection_types']:
                self.detection_stats['detection_types'][detection_type] += 1
            
            # Update daily stats
            today = datetime.now().date().isoformat()
            if today not in self.detection_stats['daily_stats']:
                self.detection_stats['daily_stats'][today] = {
                    'total_detections': 0,
                    'detection_types': {}
                }
            
            self.detection_stats['daily_stats'][today]['total_detections'] += 1
            
            if detection_type not in self.detection_stats['daily_stats'][today]['detection_types']:
                self.detection_stats['daily_stats'][today]['detection_types'][detection_type] = 0
            self.detection_stats['daily_stats'][today]['detection_types'][detection_type] += 1
            
            # Update last updated timestamp
            self.detection_stats['last_updated'] = datetime.now().isoformat()
            
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def get_recent_results(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent detection results"""
        return self.detection_results[-limit:] if self.detection_results else []
    
    def get_results_by_type(self, detection_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get detection results by type"""
        filtered_results = [r for r in self.detection_results if r['detection_type'] == detection_type]
        return filtered_results[-limit:] if filtered_results else []
    
    def get_daily_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get daily statistics for the last N days"""
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days-1)
            
            daily_stats = {}
            current_date = start_date
            
            while current_date <= end_date:
                date_str = current_date.isoformat()
                if date_str in self.detection_stats['daily_stats']:
                    daily_stats[date_str] = self.detection_stats['daily_stats'][date_str]
                else:
                    daily_stats[date_str] = {
                        'total_detections': 0,
                        'detection_types': {}
                    }
                current_date += timedelta(days=1)
            
            return daily_stats
            
        except Exception as e:
            print(f"Error getting daily stats: {e}")
            return {}
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall detection statistics"""
        return self.detection_stats.copy()
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update configuration settings"""
        try:
            self.config.update(config_updates)
            self.save_data()
            print("Configuration updated")
        except Exception as e:
            print(f"Error updating config: {e}")
    
    def get_config(self, key: Optional[str] = None) -> Any:
        """Get configuration value(s)"""
        if key is None:
            return self.config
        return self.config.get(key)
    
    def clear_old_data(self, days_to_keep: int = 30):
        """Clear old detection results"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cutoff_str = cutoff_date.isoformat()
            
            original_count = len(self.detection_results)
            self.detection_results = [
                r for r in self.detection_results 
                if r['timestamp'] >= cutoff_str
            ]
            
            removed_count = original_count - len(self.detection_results)
            print(f"Cleared {removed_count} old detection results")
            
            self.save_data()
            
        except Exception as e:
            print(f"Error clearing old data: {e}")
    
    def export_results(self, file_path: str, format: str = 'json'):
        """Export detection results to file"""
        try:
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.detection_results, f, indent=2, ensure_ascii=False)
            else:
                print(f"Unsupported export format: {format}")
                return False
            
            print(f"Results exported to {file_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False

