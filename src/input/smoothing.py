import numpy as np
from typing import Optional, List, Dict
from src.input.hand_tracker import HandLandmarks
from src.config import InteractionConfig

class LandmarkSmoother:
    """Applies Exponential Moving Average (EMA) smoothing to hand landmarks."""
    
    def __init__(self, config: InteractionConfig):
        self.config = config
        # Map handedness to previous positions
        self.previous_positions: Dict[str, np.ndarray] = {}
        
    def smooth(self, current_landmarks: List[HandLandmarks]) -> List[HandLandmarks]:
        """
        Smooth the current landmarks using EMA.
        
        Args:
            current_landmarks: Raw list of HandLandmarks.
            
        Returns:
            Smoothed List of HandLandmarks.
        """
        if not current_landmarks:
            self.previous_positions.clear()
            return []
            
        alpha = self.config.smoothing_alpha
        smoothed_hands = []
        
        current_handedness = set()
        
        for hand in current_landmarks:
            h_id = hand.handedness
            current_handedness.add(h_id)
            
            if h_id not in self.previous_positions:
                self.previous_positions[h_id] = hand.positions.copy()
                smoothed_hands.append(hand)
            else:
                prev = self.previous_positions[h_id]
                smoothed_pos = (alpha * prev) + ((1.0 - alpha) * hand.positions)
                self.previous_positions[h_id] = smoothed_pos.copy()
                
                smoothed_hands.append(HandLandmarks(
                    positions=smoothed_pos,
                    handedness=hand.handedness,
                    confidence=hand.confidence
                ))
                
        # Clean up missing hands
        for h_id in list(self.previous_positions.keys()):
            if h_id not in current_handedness:
                del self.previous_positions[h_id]
                
        return smoothed_hands
        
    def reset(self):
        """Manually reset the smoothing state (e.g. on scene reset)."""
        self.previous_positions.clear()
