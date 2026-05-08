import pytest
import os
import tempfile
from json_memory.smart import SmartMemory
from json_memory.visualizer import MemoryVisualizer

@pytest.fixture
def mem():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "test_vis.json")
        memory = SmartMemory(path)
        yield memory

def test_tree_view_basic(mem):
    mem.remember("user.name", "Alice")
    mem.remember("user.age", 30)
    mem.remember("system.version", "1.0")

    vis = MemoryVisualizer(mem)
    tree = vis.tree_view()

    assert "Memory Tree:" in tree
    assert "user/" in tree
    assert "name = Alice" in tree
    assert "age = 30" in tree
    assert "system/" in tree
    assert "version = 1.0" in tree

def test_tree_view_max_depth(mem):
    mem.remember("a.b.c.d", "deep")

    vis = MemoryVisualizer(mem)

    tree_depth_1 = vis.tree_view(max_depth=1)
    assert "a/" in tree_depth_1
    assert "b/" in tree_depth_1
    assert "c/" not in tree_depth_1

    tree_depth_2 = vis.tree_view(max_depth=2)
    assert "c/" in tree_depth_2
    assert "d" not in tree_depth_2

def test_tree_view_show_values(mem):
    mem.remember("a.b", "value123")

    vis = MemoryVisualizer(mem)

    tree_with_values = vis.tree_view(show_values=True)
    assert "value123" in tree_with_values

    tree_without_values = vis.tree_view(show_values=False)
    assert "value123" not in tree_without_values

def test_tree_view_empty(mem):
    vis = MemoryVisualizer(mem)
    tree = vis.tree_view()
    assert "Memory Tree:" in tree
    assert len(tree.split('\n')) == 2  # Header lines

def test_tree_view_long_values(mem):
    long_string = "a" * 100
    mem.remember("long.string", long_string)

    vis = MemoryVisualizer(mem)
    tree = vis.tree_view()

    # Check that the long string is truncated (default max_length=50)
    assert "a" * 47 + "..." in tree
    assert "a" * 100 not in tree

def test_tree_view_complex_values(mem):
    # Setting an entire dict at once is not tested by mem.remember as it flattens it
    # We must use mem.mem.set directly for a list that contains non-dict elements or just a list
    mem.mem.set("complex.list", [1, 2, 3])

    vis = MemoryVisualizer(mem)
    tree = vis.tree_view()

    # Lists and dicts are json dumped
    assert "[1, 2, 3]" in tree
