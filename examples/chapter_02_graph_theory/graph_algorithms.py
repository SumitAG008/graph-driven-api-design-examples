 #examples/chapter_02_graph_theory/graph_algorithms.py
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
import heapq
import random
from collections import defaultdict, deque
import asyncio
import json

@dataclass
class GraphNode:
    """Represents a node in the graph"""
    id: str
    labels: List[str]
    properties: Dict[str, any]

@dataclass 
class GraphRelationship:
    """Represents a relationship in the graph"""
    id: str
    type: str
    start_node: str
    end_node: str
    properties: Dict[str, any]

class GraphMetricsCalculator:
    """Calculate various graph metrics and centrality measures"""
    
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = {}
        self.relationships = {}
        
    def add_node(self, node: GraphNode):
        """Add a node to the graph"""
        self.nodes[node.id] = node
        
    def add_relationship(self, rel: GraphRelationship):
        """Add a relationship to the graph"""
        self.relationships[rel.id] = rel
        self.graph[rel.start_node].append((rel.end_node, rel.id, rel.properties.get('weight', 1.0)))
        
        # Add reverse for undirected relationships
        if rel.properties.get('bidirectional', False):
            self.graph[rel.end_node].append((rel.start_node, rel.id, rel.properties.get('weight', 1.0)))
    
    def calculate_degree_centrality(self) -> Dict[str, float]:
        """Calculate degree centrality for all nodes"""
        total_nodes = len(self.nodes)
        if total_nodes <= 1:
            return {node_id: 0.0 for node_id in self.nodes}
            
        centrality = {}
        for node_id in self.nodes:
            degree = len(self.graph[node_id])
            # Normalize by maximum possible degree
            centrality[node_id] = degree / (total_nodes - 1)
            
        return centrality
    
    def calculate_betweenness_centrality(self) -> Dict[str, float]:
        """Calculate betweenness centrality using Brandes algorithm"""
        centrality = {node: 0.0 for node in self.nodes}
        
        for source in self.nodes:
            # BFS to find shortest paths
            stack = []
            paths = {node: [] for node in self.nodes}
            paths[source] = [source]
            dist = {node: -1 for node in self.nodes}
            dist[source] = 0
            queue = deque([source])
            
            while queue:
                current = queue.popleft()
                stack.append(current)
                
                for neighbor, _, _ in self.graph[current]:
                    if dist[neighbor] < 0:
                        queue.append(neighbor)
                        dist[neighbor] = dist[current] + 1
                        
                    if dist[neighbor] == dist[current] + 1:
                        paths[neighbor].extend(paths[current])
            
            # Calculate centrality contribution
            dependency = {node: 0.0 for node in self.nodes}
            
            while stack:
                node = stack.pop()
                for predecessor in paths[node]:
                    if predecessor != node:
                        dependency[predecessor] += (1 + dependency[node]) / len(paths[node])
                        
                if node != source:
                    centrality[node] += dependency[node]
        
        # Normalize
        n = len(self.nodes)
        if n > 2:
            norm = 2.0 / ((n - 1) * (n - 2))
            for node in centrality:
                centrality[node] *= norm
                
        return centrality
    
    def calculate_pagerank(self, damping: float = 0.85, iterations: int = 100) -> Dict[str, float]:
        """Calculate PageRank centrality"""
        nodes = list(self.nodes.keys())
        n = len(nodes)
        
        if n == 0:
            return {}
            
        # Initialize PageRank values
        pagerank = {node: 1.0 / n for node in nodes}
        
        for _ in range(iterations):
            new_pagerank = {}
            
            for node in nodes:
                rank_sum = 0.0
                
                # Sum contributions from incoming links
                for source in nodes:
                    if any(neighbor == node for neighbor, _, _ in self.graph[source]):
                        out_degree = len(self.graph[source])
                        if out_degree > 0:
                            rank_sum += pagerank[source] / out_degree
                
                new_pagerank[node] = (1 - damping) / n + damping * rank_sum
                
            pagerank = new_pagerank
            
        return pagerank
    
    def find_shortest_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS"""
        if start not in self.nodes or end not in self.nodes:
            return None
            
        if start == end:
            return [start]
            
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor, _, _ in self.graph[current]:
                if neighbor == end:
                    return path + [neighbor]
                    
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
                    
        return None
    
    def detect_communities_simple(self) -> Dict[str, Set[str]]:
        """Simple community detection using connected components"""
        visited = set()
        communities = {}
        community_id = 0
        
        for node in self.nodes:
            if node not in visited:
                # BFS to find connected component
                community = set()
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        community.add(current)
                        
                        for neighbor, _, _ in self.graph[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                communities[f"community_{community_id}"] = community
                community_id += 1
                
        return communities

class GraphAnalyticsEngine:
    """Advanced graph analytics and pattern detection"""
    
    def __init__(self):
        self.calculator = GraphMetricsCalculator()
        
    async def analyze_graph_structure(self, nodes: List[GraphNode], 
                                    relationships: List[GraphRelationship]) -> Dict[str, any]:
        """Comprehensive graph structure analysis"""
        
        # Build the graph
        for node in nodes:
            self.calculator.add_node(node)
        for rel in relationships:
            self.calculator.add_relationship(rel)
            
        # Calculate various metrics
        degree_centrality = self.calculator.calculate_degree_centrality()
        betweenness_centrality = self.calculator.calculate_betweenness_centrality()
        pagerank = self.calculator.calculate_pagerank()
        communities = self.calculator.detect_communities_simple()
        
        # Graph statistics
        num_nodes = len(nodes)
        num_edges = len(relationships)
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        
        return {
            "structure_metrics": {
                "num_nodes": num_nodes,
                "num_edges": num_edges,
                "density": density,
                "num_communities": len(communities)
            },
            "centrality_measures": {
                "degree_centrality": degree_centrality,
                "betweenness_centrality": betweenness_centrality,
                "pagerank": pagerank
            },
            "communities": {name: list(members) for name, members in communities.items()},
            "top_influential_nodes": self._get_top_nodes(pagerank, 5),
            "key_bridges": self._get_top_nodes(betweenness_centrality, 5),
            "most_connected": self._get_top_nodes(degree_centrality, 5)
        }
    
    def _get_top_nodes(self, centrality_dict: Dict[str, float], top_n: int) -> List[Dict[str, any]]:
        """Get top N nodes by centrality measure"""
        sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
        return [
            {"node_id": node_id, "score": score} 
            for node_id, score in sorted_nodes[:top_n]
        ]

async def test_graph_algorithms():
    """Test the graph algorithms implementation"""
    print("=== Testing Graph Algorithms ===")
    
    # Create test data
    nodes = [
        GraphNode("alice", ["Person"], {"name": "Alice", "department": "Engineering"}),
        GraphNode("bob", ["Person"], {"name": "Bob", "department": "Engineering"}),
        GraphNode("charlie", ["Person"], {"name": "Charlie", "department": "Marketing"}),
        GraphNode("diana", ["Person"], {"name": "Diana", "department": "Sales"}),
        GraphNode("eve", ["Person"], {"name": "Eve", "department": "Engineering"})
    ]
    
    relationships = [
        GraphRelationship("rel1", "COLLABORATES_WITH", "alice", "bob", {"weight": 0.8}),
        GraphRelationship("rel2", "COLLABORATES_WITH", "bob", "charlie", {"weight": 0.6}),
        GraphRelationship("rel3", "COLLABORATES_WITH", "charlie", "diana", {"weight": 0.7}),
        GraphRelationship("rel4", "COLLABORATES_WITH", "alice", "eve", {"weight": 0.9}),
        GraphRelationship("rel5", "COLLABORATES_WITH", "bob", "diana", {"weight": 0.5})
    ]
    
    # Analyze the graph
    engine = GraphAnalyticsEngine()
    analysis = await engine.analyze_graph_structure(nodes, relationships)
    
    print("\nGraph Structure Analysis:")
    print(json.dumps(analysis, indent=2))
    
    # Test shortest path
    shortest_path = engine.calculator.find_shortest_path("alice", "diana")
    print(f"\nShortest path from Alice to Diana: {shortest_path}")

if __name__ == "__main__":
    asyncio.run(test_graph_algorithms())