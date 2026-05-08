import pytest
import time
import math
from json_memory.forgetting import ForgettingCurve

@pytest.fixture
def curve():
    return ForgettingCurve()

def test_calculate_strength_basic_decay(curve):
    """Test basic decay calculation over time without reinforcement."""
    initial_strength = 1.0
    current_time = time.time()

    # 0 hours passed
    strength_0h = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=current_time,
        reinforcement_count=0,
        memory_type='fact',
        current_time=current_time
    )
    assert strength_0h == initial_strength

    # 1 hour passed
    strength_1h = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=current_time - 3600,
        reinforcement_count=0,
        memory_type='fact',
        current_time=current_time
    )
    assert strength_1h < strength_0h

    # 24 hours passed
    strength_24h = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=current_time - 86400,
        reinforcement_count=0,
        memory_type='fact',
        current_time=current_time
    )
    assert strength_24h < strength_1h

def test_calculate_strength_memory_types(curve):
    """Test that different memory types decay at different rates."""
    initial_strength = 1.0
    current_time = time.time()
    last_reinforced = current_time - 86400 # 24 hours ago

    # Identity decays slowest
    strength_identity = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=last_reinforced,
        memory_type='identity',
        current_time=current_time
    )

    # Fact decays medium
    strength_fact = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=last_reinforced,
        memory_type='fact',
        current_time=current_time
    )

    # Temporary decays fastest
    strength_temporary = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=last_reinforced,
        memory_type='temporary',
        current_time=current_time
    )

    assert strength_identity > strength_fact
    assert strength_fact > strength_temporary

def test_calculate_strength_reinforcement(curve):
    """Test that reinforcements increase memory strength and stability."""
    initial_strength = 1.0
    current_time = time.time()
    last_reinforced = current_time - 86400 # 24 hours ago

    # Unreinforced
    strength_0 = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=last_reinforced,
        reinforcement_count=0,
        current_time=current_time
    )

    # Reinforced once
    strength_1 = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=last_reinforced,
        reinforcement_count=1,
        current_time=current_time
    )

    # Reinforced thrice
    strength_3 = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=last_reinforced,
        reinforcement_count=3,
        current_time=current_time
    )

    assert strength_1 > strength_0
    assert strength_3 > strength_1

def test_calculate_strength_boundaries(curve):
    """Test that calculated strength respects min_strength and max_strength bounds."""
    initial_strength = 1.0
    current_time = time.time()

    # Max bound test (heavy reinforcement)
    strength_max = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=current_time - 10, # Very recently
        reinforcement_count=100, # Extreme reinforcement
        current_time=current_time
    )
    assert strength_max <= curve.max_strength

    # Min bound test (very old unreinforced temporary memory)
    strength_min = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=current_time - 86400 * 365, # 1 year ago
        reinforcement_count=0,
        memory_type='temporary',
        current_time=current_time
    )
    assert strength_min >= curve.min_strength

def test_calculate_strength_default_time(curve, monkeypatch):
    """Test using the default current_time uses time.time()."""
    initial_strength = 1.0
    fake_now = 1000000.0
    monkeypatch.setattr(time, 'time', lambda: fake_now)

    # Explicit time
    strength_explicit = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=fake_now - 3600,
        current_time=fake_now
    )

    # Default time
    strength_implicit = curve.calculate_strength(
        initial_strength=initial_strength,
        last_reinforced=fake_now - 3600,
        current_time=None
    )

    assert math.isclose(strength_explicit, strength_implicit)
