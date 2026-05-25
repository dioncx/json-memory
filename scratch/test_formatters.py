
import os
import tempfile
from json_memory import SmartMemory

def test_llm_formatters():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "smart.json")
        sm = SmartMemory(path=path)
        
        sm.remember("user.name", "Alice")
        sm.remember("user.goal", "Build a spaceship")
        
        # Test OpenAI
        openai = sm.to_openai_messages(query="Who is Alice?")
        assert len(openai) == 1
        assert openai[0]["role"] == "system"
        assert "Alice" in openai[0]["content"]
        assert "spaceship" in openai[0]["content"]
        
        # Test Anthropic
        anthropic = sm.to_anthropic_messages(query="What is her goal?")
        assert len(anthropic) == 1
        assert anthropic[0]["role"] == "user"
        assert "spaceship" in anthropic[0]["content"]
        
        # Test Gemini
        gemini = sm.to_gemini_content(query="spaceship")
        assert len(gemini) == 1
        assert gemini[0]["role"] == "user"
        assert "parts" in gemini[0]
        assert "spaceship" in gemini[0]["parts"][0]["text"]
        
        # Test generic to_messages
        generic = sm.to_messages("openai", query="Who is Alice?")
        assert generic == openai

if __name__ == "__main__":
    test_llm_formatters()
    print("SUCCESS: LLM formatters verified!")
