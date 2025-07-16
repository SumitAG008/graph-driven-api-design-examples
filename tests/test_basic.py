"""
Basic tests for Graph API examples
"""

import pytest
import asyncio
import sys
import os

# Add examples directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'chapter_03_api_design'))

from working_examples import MockGraphDataLayer, GraphLogicLayer, GraphNode

class TestBasicFunctionality:
    """Test basic functionality without external dependencies"""
    
    def test_graph_node_creation(self):
        node = GraphNode("test_id", ["Person"], {"name": "Test User"})
        assert node.id == "test_id"
        assert node.labels == ["Person"]
        assert node.properties["name"] == "Test User"
    
    @pytest.mark.asyncio
    async def test_mock_data_layer(self):
        data_layer = MockGraphDataLayer()
        
        # Test traversal
        result = await data_layer.execute_traversal(
            start_nodes=["advait"],
            pattern="network",
            filters={},
            limit=10
        )
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_graph_logic_layer(self):
        data_layer = MockGraphDataLayer()
        logic_layer = GraphLogicLayer(data_layer)
        
        # Test employee network
        network = await logic_layer.get_employee_network("advait")
        
        assert isinstance(network, dict)
        assert "center_employee" in network
        assert "connections" in network
        assert network["center_employee"] == "advait"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
