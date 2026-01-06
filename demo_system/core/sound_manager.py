"""
Sound manager - handles audio feedback for detection events
"""

import os
import pygame
from typing import Optional

class SoundManager:
    """Sound manager for detection feedback"""
    
    def __init__(self):
        self.sounds_enabled = True
        self.sounds = {}
        self.init_sounds()
    
    def init_sounds(self):
        """Initialize sound effects"""
        try:
            # Initialize pygame mixer
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            # Load sound effects (you can add your own sound files)
            self.load_sound_effects()
            
            print("Sound manager initialized")
            
        except Exception as e:
            print(f"Failed to initialize sound manager: {e}")
            self.sounds_enabled = False
    
    def load_sound_effects(self):
        """Load sound effect files"""
        # TODO: Add your sound files to the assets directory
        # Example sound files you might want to add:
        # - detection_sound.wav (when detection occurs)
        # - error_sound.wav (when error occurs)
        # - success_sound.wav (when operation succeeds)
        
        sound_files = {
            'detection': 'assets/detection_sound.wav',
            'error': 'assets/error_sound.wav',
            'success': 'assets/success_sound.wav'
        }
        
        for sound_name, file_path in sound_files.items():
            if os.path.exists(file_path):
                try:
                    self.sounds[sound_name] = pygame.mixer.Sound(file_path)
                    print(f"Loaded sound: {sound_name}")
                except Exception as e:
                    print(f"Failed to load sound {sound_name}: {e}")
            else:
                print(f"Sound file not found: {file_path}")
    
    def play_sound(self, sound_name: str):
        """Play a sound effect"""
        if not self.sounds_enabled:
            return
        
        if sound_name in self.sounds:
            try:
                self.sounds[sound_name].play()
            except Exception as e:
                print(f"Failed to play sound {sound_name}: {e}")
        else:
            print(f"Sound not found: {sound_name}")
    
    def play_detection_sound(self):
        """Play detection sound"""
        self.play_sound('detection')
    
    def play_error_sound(self):
        """Play error sound"""
        self.play_sound('error')
    
    def play_success_sound(self):
        """Play success sound"""
        self.play_sound('success')
    
    def set_sounds_enabled(self, enabled: bool):
        """Enable or disable sounds"""
        self.sounds_enabled = enabled
        print(f"Sounds {'enabled' if enabled else 'disabled'}")
    
    def cleanup(self):
        """Clean up sound resources"""
        try:
            pygame.mixer.quit()
            print("Sound manager cleaned up")
        except Exception as e:
            print(f"Error cleaning up sound manager: {e}")

