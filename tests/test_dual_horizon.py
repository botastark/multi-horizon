# tests/test_dual_horizon.py
"""
Tests for the dual-horizon planner module.

This module contains unit tests for coverage analysis functions,
integration tests comparing strategies, and basic validation tests.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest

from dual_horizon_planner import (
    analyze_coverage_fragmentation,
    compute_revisit_cost,
    partition_field,
    compute_long_horizon_reward,
    DualHorizonPlanner,
    create_dual_horizon_planner
)
from helper import uav_position


class TestCoverageAnalysis:
    """Unit tests for coverage analysis functions."""
    
    def test_analyze_fragmentation_fully_covered(self):
        """Test fragmentation analysis with fully covered field."""
        covered_mask = np.ones((20, 20), dtype=bool)
        result = analyze_coverage_fragmentation(covered_mask)
        
        assert result['num_patches'] == 0
        assert result['total_uncovered'] == 0
        assert result['fragmentation_score'] == 0.0
    
    def test_analyze_fragmentation_no_coverage(self):
        """Test fragmentation analysis with no coverage."""
        covered_mask = np.zeros((20, 20), dtype=bool)
        result = analyze_coverage_fragmentation(covered_mask)
        
        assert result['num_patches'] == 1
        assert result['total_uncovered'] == 400
        assert len(result['patch_sizes']) == 1
        assert result['patch_sizes'][0] == 400
    
    def test_analyze_fragmentation_multiple_patches(self):
        """Test fragmentation analysis with multiple uncovered patches."""
        covered_mask = np.ones((20, 20), dtype=bool)
        # Create two separate uncovered regions
        covered_mask[2:5, 2:5] = False  # 3x3 patch = 9 cells
        covered_mask[15:18, 15:18] = False  # 3x3 patch = 9 cells
        
        result = analyze_coverage_fragmentation(covered_mask)
        
        assert result['num_patches'] == 2
        assert result['total_uncovered'] == 18
        assert len(result['patch_sizes']) == 2
        assert all(size == 9 for size in result['patch_sizes'])
        assert result['fragmentation_score'] > 0
    
    def test_analyze_fragmentation_none_input(self):
        """Test fragmentation analysis with None input."""
        result = analyze_coverage_fragmentation(None)
        
        assert result['num_patches'] == 0
        assert result['total_uncovered'] == 0
        assert result['fragmentation_score'] == 0.0


class TestRevisitCost:
    """Unit tests for revisit cost computation."""
    
    def test_revisit_cost_no_patches(self):
        """Test revisit cost with no uncovered patches."""
        uav_pos = uav_position(((10.0, 10.0), 5.0))
        uncovered_patches = {
            'num_patches': 0,
            'patch_sizes': [],
            'patch_centroids': []
        }
        
        cost = compute_revisit_cost(uav_pos, uncovered_patches)
        assert cost == 0.0
    
    def test_revisit_cost_single_patch(self):
        """Test revisit cost with a single uncovered patch."""
        uav_pos = uav_position(((0.0, 0.0), 5.0))
        uncovered_patches = {
            'num_patches': 1,
            'patch_sizes': [100],
            'patch_centroids': [(10.0, 10.0)]  # At distance sqrt(200) ~ 14.14
        }
        
        cost = compute_revisit_cost(uav_pos, uncovered_patches, uav_speed=1.0)
        assert cost > 0
    
    def test_revisit_cost_multiple_patches(self):
        """Test revisit cost increases with more patches."""
        uav_pos = uav_position(((0.0, 0.0), 5.0))
        
        single_patch = {
            'num_patches': 1,
            'patch_sizes': [50],
            'patch_centroids': [(10.0, 10.0)]
        }
        
        multiple_patches = {
            'num_patches': 2,
            'patch_sizes': [50, 50],
            'patch_centroids': [(10.0, 10.0), (20.0, 20.0)]
        }
        
        cost_single = compute_revisit_cost(uav_pos, single_patch)
        cost_multiple = compute_revisit_cost(uav_pos, multiple_patches)
        
        # More patches should have higher revisit cost
        assert cost_multiple > cost_single


class TestPartitionField:
    """Unit tests for field partitioning."""
    
    def test_partition_basic(self):
        """Test basic field partitioning."""
        belief_map = np.full((20, 20, 2), 0.5)
        tile_size = (5, 5)
        
        tile_map, metadata = partition_field(belief_map, tile_size)
        
        # Should have 16 tiles (4x4 grid)
        assert len(metadata) == 16
        assert tile_map.shape == (20, 20)
        
        # Check that all cells are assigned to a tile
        assert np.all(tile_map >= 0)
        assert np.all(tile_map < 16)
    
    def test_partition_uneven_size(self):
        """Test partitioning with non-divisible dimensions."""
        belief_map = np.full((23, 17, 2), 0.5)
        tile_size = (5, 5)
        
        tile_map, metadata = partition_field(belief_map, tile_size)
        
        # All cells should be assigned
        assert tile_map.shape == (23, 17)
        
        # Check metadata contains valid info
        for tile_id, info in metadata.items():
            assert 'bounds' in info
            assert 'cells' in info
            assert 'entropy' in info
            assert 'center' in info
    
    def test_partition_entropy_calculation(self):
        """Test that entropy is calculated correctly for tiles."""
        # Create belief map with varying probabilities
        belief_map = np.zeros((10, 10, 2))
        belief_map[:, :, 1] = 0.5  # Maximum entropy
        belief_map[:, :, 0] = 0.5
        
        tile_size = (5, 5)
        _, metadata = partition_field(belief_map, tile_size)
        
        # All tiles should have similar positive entropy
        entropies = [info['entropy'] for info in metadata.values()]
        assert all(e > 0 for e in entropies)


class TestLongHorizonReward:
    """Unit tests for long-horizon reward computation."""
    
    def test_reward_basic(self):
        """Test basic reward computation."""
        # Create a simple mock camera class
        class MockCamera:
            def x_future(self, action, x=None):
                if x is None:
                    x = uav_position(((0.0, 0.0), 5.0))
                pos = x.position
                if action == 'right':
                    return ((pos[0] + 1.0, pos[1]), x.altitude)
                return (pos, x.altitude)
            
            def get_range(self, position=None, altitude=None, index_form=False):
                return [[0, 5], [0, 5]]
        
        state = {
            'uav_pos': uav_position(((0.0, 0.0), 5.0)),
            'belief': np.full((20, 20, 2), 0.5),
            'covered_mask': np.zeros((20, 20), dtype=bool)
        }
        
        camera = MockCamera()
        weights = {'w_coverage': 1.0, 'w_fragmentation': 0.5, 'w_revisit_cost': 0.3}
        
        reward = compute_long_horizon_reward(
            state,
            ['right', 'right'],
            camera,
            weights
        )
        
        # Should get positive reward for coverage
        assert reward > 0


class TestDualHorizonPlanner:
    """Integration tests for DualHorizonPlanner class."""
    
    @pytest.fixture
    def mock_camera(self):
        """Create a mock camera for testing."""
        class MockGrid:
            x = 20
            y = 20
            length = 1
            shape = (20, 20)
            center = False
        
        class MockCamera:
            def __init__(self):
                self.grid = MockGrid()
                self.h_range = (3, 15)
                self.position = (0.0, 0.0)
                self.altitude = 5.0
                self.xy_step = 1.0
                self.h_step = 1.0
                self.x_range = [0, 20]
                self.y_range = [0, 20]
            
            def get_x(self):
                return uav_position((self.position, self.altitude))
            
            def x_future(self, action, x=None):
                if x is None:
                    x = self.get_x()
                pos = x.position
                alt = x.altitude
                
                if action == 'right' and pos[0] + self.xy_step <= 20:
                    return ((pos[0] + self.xy_step, pos[1]), alt)
                elif action == 'left' and pos[0] - self.xy_step >= 0:
                    return ((pos[0] - self.xy_step, pos[1]), alt)
                elif action == 'front' and pos[1] + self.xy_step <= 20:
                    return ((pos[0], pos[1] + self.xy_step), alt)
                elif action == 'back' and pos[1] - self.xy_step >= 0:
                    return ((pos[0], pos[1] - self.xy_step), alt)
                elif action == 'up' and alt + self.h_step <= 15:
                    return (pos, alt + self.h_step)
                elif action == 'down' and alt - self.h_step >= 3:
                    return (pos, alt - self.h_step)
                elif action == 'hover':
                    return (pos, alt)
                return None
            
            def permitted_actions(self, x):
                actions = ['hover']
                for a in ['right', 'left', 'front', 'back', 'up', 'down']:
                    if self.x_future(a, x=x) is not None:
                        actions.append(a)
                return actions
            
            def get_range(self, position=None, altitude=None, index_form=False):
                if position is None:
                    position = self.position
                if altitude is None:
                    altitude = self.altitude
                
                # Simple footprint calculation
                half_width = int(altitude / 2)
                i_min = max(0, int(position[1]) - half_width)
                i_max = min(20, int(position[1]) + half_width)
                j_min = max(0, int(position[0]) - half_width)
                j_max = min(20, int(position[0]) + half_width)
                
                if index_form:
                    return [[i_min, i_max], [j_min, j_max]]
                return [[j_min, j_max], [i_min, i_max]]
        
        return MockCamera()
    
    def test_planner_initialization(self, mock_camera):
        """Test planner initialization."""
        planner = DualHorizonPlanner(
            uav_camera=mock_camera,
            conf_dict=None,
            mcts_params={'num_iterations': 10, 'timeout': 100},
            horizon_weights={'w_coverage': 1.0, 'w_ig': 0.8}
        )
        
        assert planner.w_coverage == 1.0
        assert planner.w_ig == 0.8
        assert planner.num_iterations == 10
    
    def test_compute_blend_weights(self, mock_camera):
        """Test blend weight computation."""
        planner = DualHorizonPlanner(
            uav_camera=mock_camera,
            mcts_params={'num_iterations': 10, 'timeout': 100}
        )
        
        state = {
            'uav_pos': uav_position(((10.0, 10.0), 5.0)),
            'belief': np.full((20, 20, 2), 0.5),
            'covered_mask': np.zeros((20, 20), dtype=bool)
        }
        
        weights = planner._compute_blend_weights(state)
        
        # Weights should sum to 1
        assert abs(weights['w_short'] + weights['w_long'] - 1.0) < 1e-6
        
        # Coverage progress should be 0 initially
        assert weights['coverage_progress'] == 0.0
    
    def test_blend_weights_change_with_coverage(self, mock_camera):
        """Test that blend weights change as coverage progresses."""
        planner = DualHorizonPlanner(
            uav_camera=mock_camera,
            mcts_params={'num_iterations': 10, 'timeout': 100}
        )
        
        # Initial state with no coverage
        state_early = {
            'uav_pos': uav_position(((10.0, 10.0), 5.0)),
            'belief': np.full((20, 20, 2), 0.5),
            'covered_mask': np.zeros((20, 20), dtype=bool)
        }
        
        # State with significant coverage
        covered_late = np.ones((20, 20), dtype=bool)
        covered_late[0:5, 0:5] = False  # Small uncovered region
        
        state_late = {
            'uav_pos': uav_position(((10.0, 10.0), 5.0)),
            'belief': np.full((20, 20, 2), 0.5),
            'covered_mask': covered_late
        }
        
        weights_early = planner._compute_blend_weights(state_early)
        weights_late = planner._compute_blend_weights(state_late)
        
        # Coverage progress should be higher for late state
        assert weights_late['coverage_progress'] > weights_early['coverage_progress']


class TestIntegration:
    """Integration tests comparing strategies."""
    
    def test_factory_function(self):
        """Test the create_dual_horizon_planner factory function."""
        class MockCamera:
            pass
        
        config = {
            'mcts_params': {
                'num_iterations': 50,
                'horizon_weights': {
                    'w_coverage': 0.9,
                    'short_horizon_depth': 3
                }
            }
        }
        
        planner = create_dual_horizon_planner(
            uav_camera=MockCamera(),
            config=config
        )
        
        assert planner.w_coverage == 0.9
        assert planner.short_horizon_depth == 3
        assert planner.num_iterations == 50


# Run tests with pytest
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
