import os
import tempfile
from json_memory.smart import SmartMemory

def test_synapse_persistence():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "test_smart.json")
        mem1 = SmartMemory(path, max_chars=3000)
        
        # Add association
        mem1.link("python", ["programming", "backend"])
        print(f"Initial associations for 'python': {mem1.associate('python')}")
        
        # Force save (meta and data are auto-saved)
        # But where is the brain saved?
        
        # New instance loads from disk
        mem2 = SmartMemory(path, max_chars=3000)
        loaded_assoc = mem2.associate("python")
        print(f"Loaded associations for 'python': {loaded_assoc}")
        
        if "programming" in loaded_assoc:
            print("SUCCESS: Synapse persisted!")
        else:
            print("FAILURE: Synapse NOT persisted!")

if __name__ == "__main__":
    test_synapse_persistence()
