"""
Tests for REAP-GPT-OSS pruning module.

Run with: pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reap_gptoss.model_util import (
    ModelAttributes,
    MODEL_ATTRS,
    get_model_attrs,
)
from reap_gptoss.prune import PruningConfig, REAPPruner, PruningResult
from reap_gptoss.observer import ActivationStats, ExpertObservation


class TestModelAttributes:
    """Tests for model attribute configuration."""
    
    def test_gptoss_attrs_defined(self):
        """Verify GPT-OSS model attributes are defined."""
        assert "GptOssForCausalLM" in MODEL_ATTRS
        assert "GptOssModel" in MODEL_ATTRS
        
    def test_gptoss_attrs_complete(self):
        """Verify GPT-OSS attributes have all required fields."""
        attrs = MODEL_ATTRS["GptOssForCausalLM"]
        
        assert attrs.moe_block == "moe"
        assert attrs.gate_proj == "gate_proj"
        assert attrs.up_proj == "up_proj"
        assert attrs.down_proj == "down_proj"
        assert attrs.experts == "experts"
        assert attrs.router == "router"
        assert attrs.num_experts == "num_experts"
        assert attrs.num_experts_per_tok == "num_experts_per_tok"
        assert attrs.fused is False


class TestPruningConfig:
    """Tests for pruning configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PruningConfig()
        
        assert config.compression_ratio == 0.5
        assert config.calibration_samples == 512
        assert config.method == "reap"
        assert config.preserve_min_experts == 2
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = PruningConfig(
            compression_ratio=0.3,
            method="frequency",
            preserve_min_experts=4,
        )
        
        assert config.compression_ratio == 0.3
        assert config.method == "frequency"
        assert config.preserve_min_experts == 4


class TestActivationStats:
    """Tests for activation statistics."""
    
    def test_empty_stats(self):
        """Test stats with no activations."""
        stats = ActivationStats(layer_idx=0, expert_idx=0)
        
        assert stats.avg_gate_value == 0.0
        assert stats.avg_activation_norm == 0.0
        assert stats.reap_score == 0.0
        
    def test_stats_accumulation(self):
        """Test accumulating statistics."""
        stats = ActivationStats(layer_idx=0, expert_idx=0)
        stats.activation_count = 10
        stats.gate_value_sum = 5.0
        stats.activation_norm_sum = 20.0
        
        assert stats.avg_gate_value == 0.5
        assert stats.avg_activation_norm == 2.0
        assert stats.reap_score == 1.0  # 0.5 * 2.0


class TestExpertObservation:
    """Tests for expert observation data structure."""
    
    def test_observation_creation(self):
        """Test creating an observation."""
        obs = ExpertObservation(
            layer_idx=5,
            router_logits=torch.randn(2, 10, 32),
            router_weights=torch.randn(2, 10, 4),
            selected_experts=torch.randint(0, 32, (2, 10, 4)),
        )
        
        assert obs.layer_idx == 5
        assert obs.router_logits.shape == (2, 10, 32)
        assert obs.router_weights.shape == (2, 10, 4)
        assert obs.selected_experts.shape == (2, 10, 4)


class TestPruningResult:
    """Tests for pruning result data structure."""
    
    def test_result_creation(self):
        """Test creating a pruning result."""
        result = PruningResult(
            experts_to_prune={0: [0, 1, 2], 1: [5, 10, 15]},
            saliency_scores={0: {i: 0.1 * i for i in range(32)}},
            original_num_experts=32,
            pruned_num_experts=29,
            compression_achieved=0.09375,
        )
        
        assert len(result.experts_to_prune[0]) == 3
        assert len(result.experts_to_prune[1]) == 3
        assert result.original_num_experts == 32
        assert result.pruned_num_experts == 29


class MockMoELayer(nn.Module):
    """Mock MoE layer for testing."""
    
    def __init__(self, num_experts: int = 32, hidden_size: int = 256):
        super().__init__()
        self.experts = nn.ModuleList([
            self._create_expert(hidden_size) for _ in range(num_experts)
        ])
        self.router = nn.Linear(hidden_size, num_experts)
        
    def _create_expert(self, hidden_size: int):
        expert = nn.Module()
        expert.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
        expert.up_proj = nn.Linear(hidden_size, hidden_size * 4)
        expert.down_proj = nn.Linear(hidden_size * 4, hidden_size)
        return expert


class MockGptOssModel(nn.Module):
    """Mock GPT-OSS model for testing."""
    
    def __init__(self, num_layers: int = 4, num_experts: int = 32):
        super().__init__()
        self.config = MagicMock()
        self.config.num_experts = num_experts
        self.config.num_experts_per_tok = 4
        
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            self._create_layer(num_experts) for _ in range(num_layers)
        ])
        
    def _create_layer(self, num_experts: int):
        layer = nn.Module()
        layer.moe = MockMoELayer(num_experts)
        return layer


class TestREAPPrunerWithMock:
    """Integration tests with mock model."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock GPT-OSS model."""
        return MockGptOssModel(num_layers=4, num_experts=8)
    
    @pytest.fixture
    def pruning_config(self):
        """Create a test pruning config."""
        return PruningConfig(
            compression_ratio=0.5,
            calibration_samples=32,
            preserve_min_experts=2,
        )
    
    def test_pruner_initialization(self, mock_model, pruning_config):
        """Test pruner initialization with mock model."""
        # Need to patch get_model_attrs for mock
        with patch('reap_gptoss.prune.get_model_attrs') as mock_attrs:
            mock_attrs.return_value = MODEL_ATTRS["GptOssForCausalLM"]
            
            with patch('reap_gptoss.prune.get_moe_layers') as mock_layers:
                mock_layers.return_value = [
                    (i, layer.moe) for i, layer in enumerate(mock_model.model.layers)
                ]
                
                with patch('reap_gptoss.prune.get_num_experts') as mock_num:
                    mock_num.return_value = 8
                    
                    pruner = REAPPruner(mock_model, pruning_config)
                    assert pruner.num_experts == 8
                    assert len(pruner.moe_layers) == 4
    
    def test_select_experts_uniform(self, mock_model, pruning_config):
        """Test expert selection with uniform pruning."""
        # Create mock saliency scores
        saliency = {
            i: {j: 0.1 * j for j in range(8)}
            for i in range(4)
        }
        
        with patch('reap_gptoss.prune.get_model_attrs') as mock_attrs:
            mock_attrs.return_value = MODEL_ATTRS["GptOssForCausalLM"]
            
            with patch('reap_gptoss.prune.get_moe_layers') as mock_layers:
                mock_layers.return_value = [
                    (i, layer.moe) for i, layer in enumerate(mock_model.model.layers)
                ]
                
                with patch('reap_gptoss.prune.get_num_experts') as mock_num:
                    mock_num.return_value = 8
                    
                    pruner = REAPPruner(mock_model, pruning_config)
                    result = pruner.select_experts_to_prune(saliency)
                    
                    # With 50% compression and min 2 experts, should prune 4 per layer
                    assert result.compression_achieved > 0
                    
                    # Verify lowest scoring experts are selected
                    for layer_idx in result.experts_to_prune:
                        pruned = result.experts_to_prune[layer_idx]
                        # Lower indices have lower scores, should be pruned first
                        for expert_idx in pruned:
                            assert expert_idx < 8 - pruning_config.preserve_min_experts


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
