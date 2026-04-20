"""
Forgetting Curve — Implement Ebbinghaus forgetting curve for natural memory decay.

This module provides functionality to model how memories decay over time,
based on the Ebbinghaus forgetting curve. It helps prioritize memories
that need reinforcement and models natural memory decay.
"""

import math
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class MemoryStrength:
    """Represents the strength of a memory at a given time."""
    path: str
    value: any
    initial_strength: float  # 0.0-1.0
    current_strength: float  # 0.0-1.0
    decay_rate: float  # How fast this memory decays
    last_reinforced: float  # Timestamp of last reinforcement
    reinforcement_count: int  # How many times reinforced
    predicted_forget_time: float  # When strength will drop below threshold


class ForgettingCurve:
    """Implement Ebbinghaus forgetting curve for memory decay."""
    
    def __init__(self, base_decay_rate: float = 0.5, reinforcement_boost: float = 0.3,
                 min_strength: float = 0.1, max_strength: float = 1.0):
        """
        Args:
            base_decay_rate: Base rate of memory decay (0.0-1.0)
            reinforcement_boost: How much reinforcement boosts strength
            min_strength: Minimum strength threshold
            max_strength: Maximum strength value
        """
        self.base_decay_rate = base_decay_rate
        self.reinforcement_boost = reinforcement_boost
        self.min_strength = min_strength
        self.max_strength = max_strength
        
        # Different types of memories decay at different rates
        self.memory_type_decay_multipliers = {
            'identity': 0.1,  # Very slow decay (names, who you are)
            'skill': 0.3,     # Slow decay (how to do things)
            'fact': 0.5,      # Medium decay (facts about the world)
            'event': 0.8,     # Fast decay (specific events)
            'temporary': 1.5, # Very fast decay (temporary info)
        }
    
    def calculate_strength(self, initial_strength: float, last_reinforced: float,
                          reinforcement_count: int = 0, memory_type: str = 'fact',
                          current_time: Optional[float] = None) -> float:
        """Calculate current memory strength based on forgetting curve.
        
        Args:
            initial_strength: Initial strength when first learned (0.0-1.0)
            last_reinforced: Timestamp of last reinforcement
            reinforcement_count: Number of times reinforced
            memory_type: Type of memory ('identity', 'skill', 'fact', 'event', 'temporary')
            current_time: Current timestamp (default: now)
            
        Returns:
            Current strength (0.0-1.0)
        """
        if current_time is None:
            current_time = time.time()
        
        # Get decay multiplier for memory type
        decay_multiplier = self.memory_type_decay_multipliers.get(memory_type, 0.5)
        
        # Calculate time since last reinforcement (in hours)
        time_since_reinforcement = (current_time - last_reinforced) / 3600
        
        # Apply Ebbinghaus forgetting curve: R = e^(-t/S)
        # Where R = retention, t = time, S = stability
        stability = self._calculate_stability(initial_strength, reinforcement_count, decay_multiplier)
        
        # Calculate retention
        retention = math.exp(-time_since_reinforcement / stability)
        
        # Apply reinforcement boost
        if reinforcement_count > 0:
            reinforcement_factor = 1 + (reinforcement_count * self.reinforcement_boost)
            retention = min(retention * reinforcement_factor, self.max_strength)
        
        # Ensure within bounds
        return max(self.min_strength, min(self.max_strength, retention))
    
    def _calculate_stability(self, initial_strength: float, reinforcement_count: int,
                           decay_multiplier: float) -> float:
        """Calculate memory stability (how long it lasts).
        
        Higher stability = slower decay.
        """
        # Base stability from initial strength
        base_stability = initial_strength * 10  # 0-10 hours base
        
        # Reinforcement increases stability
        reinforcement_factor = 1 + (reinforcement_count * 0.5)
        
        # Memory type affects stability
        type_factor = 1 / decay_multiplier
        
        return base_stability * reinforcement_factor * type_factor
    
    def predict_forget_time(self, initial_strength: float, last_reinforced: float,
                           reinforcement_count: int = 0, memory_type: str = 'fact',
                           threshold: float = 0.3) -> float:
        """Predict when memory strength will drop below threshold.
        
        Args:
            initial_strength: Initial strength
            last_reinforced: Timestamp of last reinforcement
            reinforcement_count: Number of times reinforced
            memory_type: Type of memory
            threshold: Strength threshold to predict for
            
        Returns:
            Timestamp when strength will drop below threshold
        """
        if threshold <= 0:
            return float('inf')
        
        # Get decay multiplier for memory type
        decay_multiplier = self.memory_type_decay_multipliers.get(memory_type, 0.5)
        
        # Calculate stability
        stability = self._calculate_stability(initial_strength, reinforcement_count, decay_multiplier)
        
        # Solve for time: threshold = e^(-t/stability)
        # ln(threshold) = -t/stability
        # t = -stability * ln(threshold)
        time_to_threshold = -stability * math.log(threshold)
        
        return last_reinforced + (time_to_threshold * 3600)  # Convert hours to seconds
    
    def analyze_memory(self, path: str, value: any, initial_strength: float,
                      last_reinforced: float, reinforcement_count: int = 0,
                      memory_type: str = 'fact', current_time: Optional[float] = None) -> MemoryStrength:
        """Analyze a memory's strength and decay characteristics.
        
        Args:
            path: Memory path
            value: Memory value
            initial_strength: Initial strength
            last_reinforced: Timestamp of last reinforcement
            reinforcement_count: Number of times reinforced
            memory_type: Type of memory
            current_time: Current timestamp (default: now)
            
        Returns:
            MemoryStrength object with analysis
        """
        if current_time is None:
            current_time = time.time()
        
        current_strength = self.calculate_strength(
            initial_strength, last_reinforced, reinforcement_count,
            memory_type, current_time
        )
        
        predicted_forget_time = self.predict_forget_time(
            initial_strength, last_reinforced, reinforcement_count,
            memory_type, threshold=0.3
        )
        
        # Calculate decay rate (change in strength per hour)
        time_since_reinforcement = (current_time - last_reinforced) / 3600
        if time_since_reinforcement > 0:
            decay_rate = (initial_strength - current_strength) / time_since_reinforcement
        else:
            decay_rate = 0
        
        return MemoryStrength(
            path=path,
            value=value,
            initial_strength=initial_strength,
            current_strength=current_strength,
            decay_rate=decay_rate,
            last_reinforced=last_reinforced,
            reinforcement_count=reinforcement_count,
            predicted_forget_time=predicted_forget_time
        )
    
    def prioritize_for_reinforcement(self, memories: List[Dict[str, Any]], 
                                   current_time: Optional[float] = None,
                                   max_items: int = 10) -> List[Dict[str, Any]]:
        """Prioritize memories that need reinforcement.
        
        Args:
            memories: List of memory dicts with required fields
            current_time: Current timestamp (default: now)
            max_items: Maximum items to return
            
        Returns:
            List of memories prioritized by need for reinforcement
        """
        if current_time is None:
            current_time = time.time()
        
        analyzed = []
        for memory in memories:
            strength_analysis = self.analyze_memory(
                path=memory['path'],
                value=memory.get('value'),
                initial_strength=memory.get('initial_strength', 1.0),
                last_reinforced=memory.get('last_reinforced', current_time),
                reinforcement_count=memory.get('reinforcement_count', 0),
                memory_type=memory.get('memory_type', 'fact'),
                current_time=current_time
            )
            
            # Calculate reinforcement priority
            # Higher priority = lower strength + closer to forgetting
            priority = (1 - strength_analysis.current_strength) * 0.7
            time_to_forget = strength_analysis.predicted_forget_time - current_time
            if time_to_forget > 0:
                # Sooner to forget = higher priority
                priority += (1 / (time_to_forget / 3600 + 1)) * 0.3
            
            analyzed.append({
                **memory,
                'current_strength': strength_analysis.current_strength,
                'decay_rate': strength_analysis.decay_rate,
                'predicted_forget_time': strength_analysis.predicted_forget_time,
                'reinforcement_priority': priority,
            })
        
        # Sort by priority (highest first)
        analyzed.sort(key=lambda x: x['reinforcement_priority'], reverse=True)
        
        return analyzed[:max_items]
    
    def simulate_decay(self, initial_strength: float, reinforcement_count: int = 0,
                      memory_type: str = 'fact', days: int = 30) -> List[Dict[str, Any]]:
        """Simulate memory decay over time.
        
        Args:
            initial_strength: Initial strength
            reinforcement_count: Number of reinforcements
            memory_type: Type of memory
            days: Number of days to simulate
            
        Returns:
            List of daily strength values
        """
        simulation = []
        current_time = time.time()
        
        for day in range(days + 1):
            timestamp = current_time + (day * 86400)  # Add days
            strength = self.calculate_strength(
                initial_strength=initial_strength,
                last_reinforced=current_time,  # Assume just reinforced
                reinforcement_count=reinforcement_count,
                memory_type=memory_type,
                current_time=timestamp
            )
            
            simulation.append({
                'day': day,
                'timestamp': timestamp,
                'strength': strength,
                'percent_remaining': strength / initial_strength * 100 if initial_strength > 0 else 0,
            })
        
        return simulation


def create_forgetting_curve(**kwargs) -> ForgettingCurve:
    """Convenience function to create a ForgettingCurve instance."""
    return ForgettingCurve(**kwargs)


# Test the forgetting curve
if __name__ == "__main__":
    curve = ForgettingCurve()
    
    # Test strength calculation
    print("=== Strength Calculation ===")
    current_time = time.time()
    
    # Memory reinforced 1 hour ago
    strength = curve.calculate_strength(
        initial_strength=1.0,
        last_reinforced=current_time - 3600,  # 1 hour ago
        reinforcement_count=0,
        memory_type='fact'
    )
    print(f"Fact memory (1 hour old): {strength:.3f}")
    
    # Memory reinforced 24 hours ago
    strength = curve.calculate_strength(
        initial_strength=1.0,
        last_reinforced=current_time - 86400,  # 24 hours ago
        reinforcement_count=0,
        memory_type='fact'
    )
    print(f"Fact memory (24 hours old): {strength:.3f}")
    
    # Reinforced memory
    strength = curve.calculate_strength(
        initial_strength=1.0,
        last_reinforced=current_time - 86400,  # 24 hours ago
        reinforcement_count=3,
        memory_type='fact'
    )
    print(f"Reinforced memory (24 hours, 3 reinforcements): {strength:.3f}")
    
    # Identity memory (slow decay)
    strength = curve.calculate_strength(
        initial_strength=1.0,
        last_reinforced=current_time - 86400 * 7,  # 7 days ago
        reinforcement_count=0,
        memory_type='identity'
    )
    print(f"Identity memory (7 days old): {strength:.3f}")
    
    # Test simulation
    print("\n=== Decay Simulation (30 days) ===")
    simulation = curve.simulate_decay(
        initial_strength=1.0,
        reinforcement_count=0,
        memory_type='fact',
        days=30
    )
    
    for day_data in [simulation[0], simulation[7], simulation[14], simulation[29]]:
        print(f"Day {day_data['day']}: {day_data['strength']:.3f} ({day_data['percent_remaining']:.1f}%)")
    
    # Test prioritization
    print("\n=== Prioritization ===")
    memories = [
        {'path': 'user.name', 'value': 'Dion', 'memory_type': 'identity', 'last_reinforced': current_time - 86400 * 30},
        {'path': 'project.status', 'value': 'active', 'memory_type': 'fact', 'last_reinforced': current_time - 86400},
        {'path': 'meeting.time', 'value': '3pm', 'memory_type': 'event', 'last_reinforced': current_time - 3600},
    ]
    
    prioritized = curve.prioritize_for_reinforcement(memories, current_time)
    for item in prioritized:
        print(f"{item['path']}: strength={item['current_strength']:.3f}, priority={item['reinforcement_priority']:.3f}")