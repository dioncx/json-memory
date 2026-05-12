import os
import tempfile
from json_memory.memory import Memory

def test_history_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "test_mem.json")
        mem1 = Memory(auto_flush_path=path, track_history=True)
        
        mem1.set("user.name", "Alice")
        print(f"Initial history entries: {len(mem1.history())}")
        
        history_file = path.replace(".json", ".history.json")
        print(f"History file path: {history_file}")
        print(f"History file exists: {os.path.exists(history_file)}")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                print(f"History file content: {f.read()}")
        
        # New instance
        mem2 = Memory(auto_flush_path=path, track_history=True)
        print(f"Loaded history entries: {len(mem2.history())}")
        
        if len(mem2.history()) > 0:
            print("SUCCESS: History persisted!")
        else:
            print("FAILURE: History NOT persisted!")

if __name__ == "__main__":
    test_history_persistence()
