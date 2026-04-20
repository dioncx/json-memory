"""Tests for procedural memory and skill transfer."""
import json
import os
import tempfile
import time

import pytest

from json_memory.smart import SmartMemory, ProceduralMemory, Skill


class TestSkill:
    """Test Skill class."""
    
    def test_skill_creation(self):
        """Skill should be created with correct attributes."""
        skill = Skill(
            name="balance",
            principle="Forward momentum stabilizes lateral movement",
            domains=["cycling", "motorcycling"]
        )
        
        assert skill.name == "balance"
        assert skill.principle == "Forward momentum stabilizes lateral movement"
        assert skill.domains == ["cycling", "motorcycling"]
        assert skill.strength == 1.0
        assert skill.transfer_count == 0
    
    def test_skill_defaults(self):
        """Skill should have sensible defaults."""
        skill = Skill(name="test", principle="test principle")
        
        assert skill.domains == []
        assert skill.strength == 1.0
        assert skill.examples == []
        assert skill.transfer_count == 0


class TestProceduralMemory:
    """Test ProceduralMemory class."""
    
    @pytest.fixture
    def proc_mem(self):
        """Create a temporary ProceduralMemory instance."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            memory = ProceduralMemory(path=temp_path)
            yield memory
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_add_skill(self, proc_mem):
        """Should add skill to memory."""
        skill = proc_mem.add_skill(
            name="balance",
            principle="Forward momentum stabilizes lateral movement",
            domains=["cycling"],
            examples=["Learned to ride bicycle at age 8"]
        )
        
        assert "balance" in proc_mem.skills
        assert skill.principle == "Forward momentum stabilizes lateral movement"
        assert skill.domains == ["cycling"]
        assert len(skill.examples) == 1
    
    def test_strengthen_existing_skill(self, proc_mem):
        """Adding same skill should strengthen it."""
        # Add skill first time with lower initial strength
        skill1 = proc_mem.add_skill(
            name="balance",
            principle="Forward momentum stabilizes lateral movement",
            domains=["cycling"]
        )
        # Manually set lower strength to test strengthening
        skill1.strength = 0.8
        initial_strength = skill1.strength
        
        # Add same skill again
        skill2 = proc_mem.add_skill(
            name="balance",
            principle="Forward momentum stabilizes lateral movement",
            domains=["motorcycling"],
            examples=["Applied to motorcycle riding"]
        )
        
        # Should be same skill, strengthened
        assert skill1 is skill2
        assert skill2.strength > initial_strength
        assert "motorcycling" in skill2.domains
        assert len(skill2.examples) == 1
    
    def test_get_skills_for_domain(self, proc_mem):
        """Should get all skills for a domain."""
        proc_mem.add_skill("balance", "principle1", domains=["cycling"])
        proc_mem.add_skill("momentum", "principle2", domains=["cycling", "motorcycling"])
        proc_mem.add_skill("steering", "principle3", domains=["motorcycling"])
        
        cycling_skills = proc_mem.get_skills_for_domain("cycling")
        assert len(cycling_skills) == 2
        assert any(s.name == "balance" for s in cycling_skills)
        assert any(s.name == "momentum" for s in cycling_skills)
    
    def test_find_transferable_skills(self, proc_mem):
        """Should find skills that transfer to new domain."""
        proc_mem.add_skill(
            "balance",
            "Forward momentum stabilizes lateral movement",
            domains=["cycling"]
        )
        
        # Find skills for motorcycling (similar domain)
        transferable = proc_mem.find_transferable_skills("motorcycling")
        assert len(transferable) == 1
        assert transferable[0].name == "balance"
    
    def test_find_transferable_by_keywords(self, proc_mem):
        """Should find skills by keyword overlap."""
        proc_mem.add_skill(
            "balance",
            "Forward momentum stabilizes lateral movement",
            domains=["cycling"]
        )
        
        # Find by keywords
        transferable = proc_mem.find_transferable_skills(
            "motorcycling",
            context_keywords={"momentum", "stabilizes"}
        )
        assert len(transferable) == 1
        assert transferable[0].name == "balance"
    
    def test_apply_skill(self, proc_mem):
        """Applying skill should strengthen it."""
        skill = proc_mem.add_skill(
            "balance",
            "Forward momentum stabilizes lateral movement",
            domains=["cycling"]
        )
        # Set lower initial strength to test strengthening
        skill.strength = 0.7
        initial_strength = skill.strength
        initial_transfer = skill.transfer_count
        
        # Apply skill
        success = proc_mem.apply_skill("balance", "motorcycling")
        
        assert success is True
        assert skill.strength > initial_strength
        assert skill.transfer_count == initial_transfer + 1
        assert "motorcycling" in skill.domains
    
    def test_apply_nonexistent_skill(self, proc_mem):
        """Applying nonexistent skill should return False."""
        success = proc_mem.apply_skill("nonexistent")
        assert success is False
    
    def test_extract_principles(self, proc_mem):
        """Should extract principles from experience text."""
        experience = "Learned that forward momentum causes balance stability"
        principles = proc_mem.extract_principles(experience, domain="cycling")
        
        assert len(principles) == 1
        assert principles[0]['type'] == 'causal'
        assert 'forward momentum' in principles[0]['principle']
        assert 'balance stability' in principles[0]['principle']
    
    def test_competence_map(self, proc_mem):
        """Should return overview of skills."""
        proc_mem.add_skill("balance", "principle1", domains=["cycling"])
        proc_mem.add_skill("momentum", "principle2", domains=["cycling", "motorcycling"])
        
        competence = proc_mem.competence_map()
        
        assert competence['total_skills'] == 2
        assert 'cycling' in competence['domains']
        assert 'motorcycling' in competence['domains']
        assert len(competence['strongest']) == 2
    
    def test_save_and_load(self, proc_mem):
        """Skills should persist across save/load."""
        # Add skill
        proc_mem.add_skill(
            "balance",
            "Forward momentum stabilizes lateral movement",
            domains=["cycling"],
            examples=["Learned at age 8"]
        )
        
        # Save
        proc_mem._save()
        
        # Create new instance and load
        new_mem = ProceduralMemory(path=proc_mem.path)
        
        assert "balance" in new_mem.skills
        skill = new_mem.skills["balance"]
        assert skill.principle == "Forward momentum stabilizes lateral movement"
        assert skill.domains == ["cycling"]
        assert skill.examples == ["Learned at age 8"]


class TestSmartMemoryProcedural:
    """Test SmartMemory with procedural memory enabled."""
    
    @pytest.fixture
    def mem(self):
        """Create SmartMemory with procedural memory enabled."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            memory = SmartMemory(
                path=temp_path,
                max_chars=5000,
                procedural=True
            )
            yield memory
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_learn_extracts_principles(self, mem):
        """learn() should extract principles from experience."""
        result = mem.learn(
            experience="Learned that forward momentum causes balance stability",
            domain="cycling"
        )
        
        assert 'principles_extracted' in result
        assert len(result['principles_extracted']) > 0
        assert 'skills_created' in result
    
    def test_transfer_finds_skills(self, mem):
        """transfer() should find applicable skills."""
        # First, learn something
        mem.learn(
            experience="Learned that forward momentum causes balance stability",
            domain="cycling"
        )
        
        # Then, check transfer to new domain
        result = mem.transfer(
            new_situation="Need to balance a motorcycle",
            domain="motorcycling"
        )
        
        assert 'transferable_skills' in result
        assert len(result['transferable_skills']) > 0
    
    def test_apply_skill_records_application(self, mem):
        """apply_skill() should record skill application."""
        # First, learn something
        learn_result = mem.learn(
            experience="Learned that forward momentum causes balance stability",
            domain="cycling"
        )
        
        skill_name = learn_result['skills_created'][0]
        
        # Apply the skill
        success = mem.apply_skill(
            skill_name=skill_name,
            new_domain="motorcycling",
            outcome="Successfully balanced motorcycle using same principle"
        )
        
        assert success is True
    
    def test_competence_map_shows_skills(self, mem):
        """competence_map() should show all skills."""
        # Learn something
        mem.learn(
            experience="Learned that forward momentum causes balance stability",
            domain="cycling"
        )
        
        # Get competence map
        competence = mem.competence_map()
        
        assert 'total_skills' in competence
        assert competence['total_skills'] > 0
        assert 'domains' in competence
    
    def test_learn_without_procedural_fails(self):
        """learn() should fail if procedural memory not enabled."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({}, f)
            temp_path = f.name
        
        try:
            memory = SmartMemory(path=temp_path, max_chars=5000, procedural=False)
            
            result = memory.learn(
                experience="Test experience",
                domain="test"
            )
            
            assert 'error' in result
            assert 'Procedural memory not enabled' in result['error']
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
