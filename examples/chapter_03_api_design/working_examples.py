"""
Working examples for Graph-Driven API Design book
These examples are tested and functional
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import json

# Chapter 3: Core Graph API Components
@dataclass
class GraphNode:
    id: str
    labels: List[str]
    properties: Dict[str, Any]

@dataclass 
class GraphRelationship:
    id: str
    type: str
    start_node: str
    end_node: str
    properties: Dict[str, Any]

@dataclass
class GraphPath:
    nodes: List[GraphNode]
    relationships: List[GraphRelationship]
    length: int

class GraphDataLayer(ABC):
    """Abstract interface for graph data operations"""
    
    @abstractmethod
    async def execute_traversal(
        self, 
        start_nodes: List[str], 
        pattern: str, 
        filters: Dict[str, Any], 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Execute graph traversal query"""
        pass
    
    @abstractmethod
    async def find_shortest_path(
        self, 
        start_id: str, 
        end_id: str, 
        max_depth: int = 6
    ) -> Optional[GraphPath]:
        """Find shortest path between two nodes"""
        pass

class MockGraphDataLayer(GraphDataLayer):
    """Mock implementation for testing"""
    
    def __init__(self):
        # Sample data for testing
        self.nodes = {
            "advait": GraphNode("advait", ["Person"], {"name": "Advait Sharma", "title": "Senior Developer"}),
            "priya": GraphNode("priya", ["Person"], {"name": "Priya Patel", "title": "Tech Lead"}),
            "project_alpha": GraphNode("project_alpha", ["Project"], {"name": "Alpha Release", "status": "Active"})
        }
        
        self.relationships = [
            GraphRelationship("rel1", "REPORTS_TO", "advait", "priya", {"since": "2023-01-01"}),
            GraphRelationship("rel2", "WORKS_ON", "advait", "project_alpha", {"role": "Developer", "allocation": 0.8})
        ]
    
    async def execute_traversal(
        self, 
        start_nodes: List[str], 
        pattern: str, 
        filters: Dict[str, Any], 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Mock traversal implementation"""
        results = []
        
        for start_id in start_nodes:
            if start_id in self.nodes:
                # Find connected nodes
                connected_rels = [rel for rel in self.relationships 
                                if rel.start_node == start_id or rel.end_node == start_id]
                
                for rel in connected_rels:
                    connected_id = rel.end_node if rel.start_node == start_id else rel.start_node
                    if connected_id in self.nodes:
                        results.append({
                            "node": self.nodes[connected_id].__dict__,
                            "relationship": rel.__dict__,
                            "distance": 1
                        })
                        
        return results[:limit]
    
    async def find_shortest_path(
        self, 
        start_id: str, 
        end_id: str, 
        max_depth: int = 6
    ) -> Optional[GraphPath]:
        """Mock shortest path implementation"""
        if start_id == "advait" and end_id == "priya":
            return GraphPath(
                nodes=[self.nodes["advait"], self.nodes["priya"]],
                relationships=[self.relationships[0]],  # REPORTS_TO relationship
                length=1
            )
        return None

# Chapter 4: Graph Logic Layer
class GraphLogicLayer:
    """Business logic layer for graph operations"""
    
    def __init__(self, data_layer: GraphDataLayer):
        self.data_layer = data_layer
    
    async def get_employee_network(
        self, 
        employee_id: str, 
        depth: int = 2, 
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get network around an employee"""
        
        filters = {}
        if relationship_types:
            filters["relationship_types"] = relationship_types
            
        connections = await self.data_layer.execute_traversal(
            start_nodes=[employee_id],
            pattern="network",
            filters=filters,
            limit=50
        )
        
        return {
            "center_employee": employee_id,
            "connections": connections,
            "total_connections": len(connections),
            "max_depth": depth
        }
    
    async def find_collaboration_path(
        self, 
        person1_id: str, 
        person2_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find how two people are connected through collaboration"""
        
        path = await self.data_layer.find_shortest_path(person1_id, person2_id)
        
        if path:
            return {
                "path_exists": True,
                "path_length": path.length,
                "connection_strength": self._calculate_path_strength(path),
                "path_details": {
                    "nodes": [node.__dict__ for node in path.nodes],
                    "relationships": [rel.__dict__ for rel in path.relationships]
                }
            }
        
        return {"path_exists": False}
    
    def _calculate_path_strength(self, path: GraphPath) -> float:
        """Calculate the strength of a connection path"""
        if not path.relationships:
            return 0.0
            
        # Simple calculation based on relationship types and recency
        base_strength = 1.0 / path.length  # Shorter paths are stronger
        
        # Boost for direct management relationships
        for rel in path.relationships:
            if rel.type in ["REPORTS_TO", "MANAGES"]:
                base_strength *= 1.5
            elif rel.type in ["COLLABORATES_WITH", "WORKS_ON"]:
                base_strength *= 1.2
                
        return min(base_strength, 1.0)

# Example usage and testing
async def main():
    """Test the graph API components"""
    
    # Initialize components
    data_layer = MockGraphDataLayer()
    logic_layer = GraphLogicLayer(data_layer)
    
    print("=== Testing Graph API Components ===\n")
    
    # Test 1: Get employee network
    print("Test 1: Employee Network")
    network = await logic_layer.get_employee_network("advait", depth=2)
    print(f"Network for advait: {json.dumps(network, indent=2, default=str)}\n")
    
    # Test 2: Find collaboration path
    print("Test 2: Collaboration Path")
    path = await logic_layer.find_collaboration_path("advait", "priya")
    print(f"Path from advait to priya: {json.dumps(path, indent=2, default=str)}\n")
    
    # Test 3: Direct traversal
    print("Test 3: Direct Traversal")
    traversal_result = await data_layer.execute_traversal(
        start_nodes=["advait"],
        pattern="network",
        filters={},
        limit=10
    )
    print(f"Traversal from advait: {json.dumps(traversal_result, indent=2, default=str)}")

if __name__ == "__main__":
    asyncio.run(main())