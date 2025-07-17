#examples/chapter_05_query_languages/traversal_patterns.py

from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import asyncio
import json
import time

class TraversalStrategy(Enum):
    """Different traversal strategies"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    SHORTEST_PATH = "shortest_path"
    ALL_PATHS = "all_paths"
    WEIGHTED_SHORTEST = "weighted_shortest"

@dataclass
class GraphNode:
    """Graph node representation"""
    id: str
    labels: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GraphEdge:
    """Graph edge representation"""
    id: str
    from_node: str
    to_node: str
    edge_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

@dataclass
class TraversalPath:
    """Represents a path through the graph"""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    total_weight: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryFilter:
    """Filter criteria for graph queries"""
    node_labels: Optional[List[str]] = None
    edge_types: Optional[List[str]] = None
    property_filters: Optional[Dict[str, Any]] = None
    max_depth: int = 5
    limit: Optional[int] = None

class CypherQueryBuilder:
    """Build Cypher-like queries for graph traversal"""
    
    def __init__(self):
        self.query_parts = []
        self.parameters = {}
        
    def match(self, pattern: str) -> 'CypherQueryBuilder':
        """Add a MATCH clause"""
        self.query_parts.append(f"MATCH {pattern}")
        return self
        
    def where(self, condition: str) -> 'CypherQueryBuilder':
        """Add a WHERE clause"""
        self.query_parts.append(f"WHERE {condition}")
        return self
        
    def return_clause(self, items: str) -> 'CypherQueryBuilder':
        """Add a RETURN clause"""
        self.query_parts.append(f"RETURN {items}")
        return self
        
    def limit(self, count: int) -> 'CypherQueryBuilder':
        """Add a LIMIT clause"""
        self.query_parts.append(f"LIMIT {count}")
        return self
        
    def order_by(self, field: str, direction: str = "ASC") -> 'CypherQueryBuilder':
        """Add an ORDER BY clause"""
        self.query_parts.append(f"ORDER BY {field} {direction}")
        return self
        
    def build(self) -> str:
        """Build the final query string"""
        return "\n".join(self.query_parts)

class GraphTraversalEngine:
    """Advanced graph traversal engine with multiple strategies"""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.adjacency_list: Dict[str, List[GraphEdge]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)
        
    def add_node(self, node: GraphNode):
        """Add a node to the graph"""
        self.nodes[node.id] = node
        
    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph"""
        self.edges[edge.id] = edge
        self.adjacency_list[edge.from_node].append(edge)
        self.reverse_adjacency[edge.to_node].append(edge)
        
    async def traverse(self, start_node_id: str, strategy: TraversalStrategy,
                      filters: QueryFilter) -> List[TraversalPath]:
        """Main traversal method with different strategies"""
        
        if start_node_id not in self.nodes:
            return []
            
        if strategy == TraversalStrategy.BREADTH_FIRST:
            return await self._breadth_first_traversal(start_node_id, filters)
        elif strategy == TraversalStrategy.DEPTH_FIRST:
            return await self._depth_first_traversal(start_node_id, filters)
        elif strategy == TraversalStrategy.SHORTEST_PATH:
            return await self._shortest_path_traversal(start_node_id, filters)
        elif strategy == TraversalStrategy.WEIGHTED_SHORTEST:
            return await self._weighted_shortest_path(start_node_id, filters)
        else:
            raise ValueError(f"Unsupported traversal strategy: {strategy}")
    
    async def _breadth_first_traversal(self, start_id: str, filters: QueryFilter) -> List[TraversalPath]:
        """Breadth-first traversal implementation"""
        visited = set()
        queue = deque([(start_id, [], [], 0)])  # (node_id, path_nodes, path_edges, depth)
        results = []
        
        while queue and len(results) < (filters.limit or float('inf')):
            current_id, path_nodes, path_edges, depth = queue.popleft()
            
            if current_id in visited or depth > filters.max_depth:
                continue
                
            visited.add(current_id)
            current_node = self.nodes[current_id]
            
            # Add current node to path
            new_path_nodes = path_nodes + [current_node]
            
            # Create traversal path for current position
            if depth > 0:  # Don't include start node alone
                path = TraversalPath(
                    nodes=new_path_nodes,
                    edges=path_edges,
                    metadata={"depth": depth, "strategy": "breadth_first"}
                )
                results.append(path)
            
            # Add neighbors to queue
            for edge in self.adjacency_list[current_id]:
                if self._edge_matches_filter(edge, filters):
                    next_node_id = edge.to_node
                    if next_node_id not in visited:
                        new_path_edges = path_edges + [edge]
                        queue.append((next_node_id, new_path_nodes, new_path_edges, depth + 1))
                        
        return results
    
    async def _depth_first_traversal(self, start_id: str, filters: QueryFilter) -> List[TraversalPath]:
        """Depth-first traversal implementation"""
        visited = set()
        stack = [(start_id, [], [], 0)]  # (node_id, path_nodes, path_edges, depth)
        results = []
        
        while stack and len(results) < (filters.limit or float('inf')):
            current_id, path_nodes, path_edges, depth = stack.pop()
            
            if current_id in visited or depth > filters.max_depth:
                continue
                
            visited.add(current_id)
            current_node = self.nodes[current_id]
            
            # Add current node to path
            new_path_nodes = path_nodes + [current_node]
            
            # Create traversal path for current position
            if depth > 0:  # Don't include start node alone
                path = TraversalPath(
                    nodes=new_path_nodes,
                    edges=path_edges,
                    metadata={"depth": depth, "strategy": "depth_first"}
                )
                results.append(path)
            
            # Add neighbors to stack (in reverse order for proper DFS)
            neighbors = []
            for edge in self.adjacency_list[current_id]:
                if self._edge_matches_filter(edge, filters):
                    next_node_id = edge.to_node
                    if next_node_id not in visited:
                        new_path_edges = path_edges + [edge]
                        neighbors.append((next_node_id, new_path_nodes, new_path_edges, depth + 1))
            
            # Add in reverse order for proper DFS ordering
            for neighbor in reversed(neighbors):
                stack.append(neighbor)
                        
        return results
    
    async def _shortest_path_traversal(self, start_id: str, filters: QueryFilter) -> List[TraversalPath]:
        """Find shortest paths using BFS"""
        if not filters.property_filters or 'target_node' not in filters.property_filters:
            # If no target specified, return BFS traversal
            return await self._breadth_first_traversal(start_id, filters)
            
        target_id = filters.property_filters['target_node']
        if target_id not in self.nodes:
            return []
            
        # BFS to find shortest path
        queue = deque([(start_id, [self.nodes[start_id]], [])])
        visited = {start_id}
        
        while queue:
            current_id, path_nodes, path_edges = queue.popleft()
            
            if current_id == target_id:
                return [TraversalPath(
                    nodes=path_nodes,
                    edges=path_edges,
                    metadata={"strategy": "shortest_path", "target": target_id}
                )]
            
            for edge in self.adjacency_list[current_id]:
                if self._edge_matches_filter(edge, filters):
                    next_id = edge.to_node
                    if next_id not in visited:
                        visited.add(next_id)
                        new_path_nodes = path_nodes + [self.nodes[next_id]]
                        new_path_edges = path_edges + [edge]
                        queue.append((next_id, new_path_nodes, new_path_edges))
        
        return []  # No path found
    
    async def _weighted_shortest_path(self, start_id: str, filters: QueryFilter) -> List[TraversalPath]:
        """Dijkstra's algorithm for weighted shortest paths"""
        if not filters.property_filters or 'target_node' not in filters.property_filters:
            return []
            
        target_id = filters.property_filters['target_node']
        if target_id not in self.nodes:
            return []
        
        # Dijkstra's algorithm
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[start_id] = 0
        previous = {}
        previous_edges = {}
        unvisited = set(self.nodes.keys())
        
        while unvisited:
            # Find unvisited node with minimum distance
            current = min(unvisited, key=lambda x: distances[x])
            if distances[current] == float('inf'):
                break
                
            unvisited.remove(current)
            
            if current == target_id:
                # Reconstruct path
                path_nodes = []
                path_edges = []
                node = target_id
                
                while node != start_id:
                    path_nodes.insert(0, self.nodes[node])
                    if node in previous_edges:
                        path_edges.insert(0, previous_edges[node])
                    node = previous[node]
                
                path_nodes.insert(0, self.nodes[start_id])
                
                return [TraversalPath(
                    nodes=path_nodes,
                    edges=path_edges,
                    total_weight=distances[target_id],
                    metadata={"strategy": "weighted_shortest", "target": target_id}
                )]
            
            # Update distances to neighbors
            for edge in self.adjacency_list[current]:
                if self._edge_matches_filter(edge, filters):
                    neighbor = edge.to_node
                    if neighbor in unvisited:
                        alt_distance = distances[current] + edge.weight
                        if alt_distance < distances[neighbor]:
                            distances[neighbor] = alt_distance
                            previous[neighbor] = current
                            previous_edges[neighbor] = edge
        
        return []  # No path found
    
    def _edge_matches_filter(self, edge: GraphEdge, filters: QueryFilter) -> bool:
        """Check if an edge matches the filter criteria"""
        if filters.edge_types and edge.edge_type not in filters.edge_types:
            return False
            
        if filters.property_filters:
            for prop, value in filters.property_filters.items():
                if prop not in ['target_node']:  # Skip special filter properties
                    if prop not in edge.properties or edge.properties[prop] != value:
                        return False
                        
        return True
    
    def _node_matches_filter(self, node: GraphNode, filters: QueryFilter) -> bool:
        """Check if a node matches the filter criteria"""
        if filters.node_labels:
            if not any(label in node.labels for label in filters.node_labels):
                return False
                
        if filters.property_filters:
            for prop, value in filters.property_filters.items():
                if prop not in ['target_node']:  # Skip special filter properties
                    if prop not in node.properties or node.properties[prop] != value:
                        return False
                        
        return True

class QueryOptimizer:
    """Optimize graph queries for performance"""
    
    def __init__(self):
        self.index_hints = {}
        self.statistics = {}
        
    async def analyze_query_cost(self, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the cost of executing a query"""
        
        base_cost = 1
        traversal_cost = 1
        filter_cost = 1
        
        # Calculate traversal cost based on depth and strategy
        max_depth = query_plan.get('max_depth', 3)
        strategy = query_plan.get('strategy', TraversalStrategy.BREADTH_FIRST)
        
        if strategy == TraversalStrategy.BREADTH_FIRST:
            traversal_cost = max_depth ** 2  # Quadratic growth
        elif strategy == TraversalStrategy.DEPTH_FIRST:
            traversal_cost = max_depth * 1.5  # Linear with overhead
        elif strategy == TraversalStrategy.SHORTEST_PATH:
            traversal_cost = max_depth * 2  # BFS with path reconstruction
        elif strategy == TraversalStrategy.WEIGHTED_SHORTEST:
            traversal_cost = max_depth ** 2 * 2  # Dijkstra complexity
        
        # Calculate filter cost
        filters = query_plan.get('filters', {})
        if filters.get('property_filters'):
            filter_cost = len(filters['property_filters']) * 0.5
        if filters.get('node_labels'):
            filter_cost += len(filters['node_labels']) * 0.3
        if filters.get('edge_types'):
            filter_cost += len(filters['edge_types']) * 0.3
            
        total_cost = base_cost * traversal_cost * filter_cost
        
        return {
            "total_cost": total_cost,
            "traversal_cost": traversal_cost,
            "filter_cost": filter_cost,
            "estimated_results": min(100, max_depth * 10),  # Rough estimate
            "optimization_suggestions": self._get_optimization_suggestions(query_plan, total_cost)
        }
    
    def _get_optimization_suggestions(self, query_plan: Dict[str, Any], cost: float) -> List[str]:
        """Provide optimization suggestions for high-cost queries"""
        suggestions = []
        
        if cost > 100:
            suggestions.append("Consider reducing max_depth to improve performance")
            
        if query_plan.get('max_depth', 3) > 5:
            suggestions.append("Deep traversals can be expensive - consider pagination")
            
        if not query_plan.get('filters', {}).get('edge_types'):
            suggestions.append("Adding edge type filters can significantly improve performance")
            
        if query_plan.get('strategy') == TraversalStrategy.WEIGHTED_SHORTEST and cost > 50:
            suggestions.append("Weighted shortest path is expensive - consider using simple shortest path if weights aren't critical")
            
        return suggestions

class AdvancedQueryPatterns:
    """Advanced query patterns and algorithms"""
    
    def __init__(self, traversal_engine: GraphTraversalEngine):
        self.engine = traversal_engine
        
    async def find_collaboration_patterns(self, person_id: str) -> Dict[str, Any]:
        """Find collaboration patterns around a person"""
        
        # Find direct collaborators
        direct_filters = QueryFilter(
            edge_types=["COLLABORATES_WITH", "WORKS_ON"],
            max_depth=1,
            limit=20
        )
        
        direct_collaborations = await self.engine.traverse(
            person_id, TraversalStrategy.BREADTH_FIRST, direct_filters
        )
        
        # Find triangle patterns (mutual collaborators)
        triangles = []
        collaborators = set()
        
        for path in direct_collaborations:
            if len(path.nodes) > 1:
                collaborator_id = path.nodes[-1].id
                collaborators.add(collaborator_id)
        
        # Check for triangular relationships
        for collaborator in collaborators:
            mutual_filters = QueryFilter(
                edge_types=["COLLABORATES_WITH"],
                max_depth=1
            )
            
            mutual_paths = await self.engine.traverse(
                collaborator, TraversalStrategy.BREADTH_FIRST, mutual_filters
            )
            
            for path in mutual_paths:
                if len(path.nodes) > 1 and path.nodes[-1].id in collaborators:
                    triangles.append({
                        "person": person_id,
                        "collaborator1": collaborator,
                        "collaborator2": path.nodes[-1].id
                    })
        
        return {
            "person_id": person_id,
            "direct_collaborators": len(collaborators),
            "collaboration_triangles": len(triangles),
            "triangle_details": triangles[:10],  # Limit results
            "network_density": len(triangles) / max(len(collaborators), 1)
        }
    
    async def find_influence_paths(self, from_person: str, to_person: str) -> List[Dict[str, Any]]:
        """Find paths of influence between two people"""
        
        influence_filters = QueryFilter(
            edge_types=["MANAGES", "MENTORS", "INFLUENCES", "REPORTS_TO"],
            max_depth=4,
            property_filters={"target_node": to_person}
        )
        
        paths = await self.engine.traverse(
            from_person, TraversalStrategy.SHORTEST_PATH, influence_filters
        )
        
        influence_paths = []
        for path in paths:
            # Calculate influence strength along the path
            influence_strength = 1.0
            for edge in path.edges:
                edge_strength = edge.properties.get("strength", 0.5)
                influence_strength *= edge_strength
            
            influence_paths.append({
                "path_length": len(path.nodes) - 1,
                "influence_strength": influence_strength,
                "path_nodes": [node.id for node in path.nodes],
                "relationship_types": [edge.edge_type for edge in path.edges]
            })
        
        return influence_paths
    
    async def detect_communities(self, seed_nodes: List[str]) -> Dict[str, Any]:
        """Detect communities around seed nodes"""
        
        communities = {}
        
        for seed in seed_nodes:
            community_filters = QueryFilter(
                edge_types=["COLLABORATES_WITH", "FRIENDS_WITH", "SAME_TEAM"],
                max_depth=2,
                limit=50
            )
            
            community_paths = await self.engine.traverse(
                seed, TraversalStrategy.BREADTH_FIRST, community_filters
            )
            
            # Collect all nodes in this community
            community_nodes = {seed}
            for path in community_paths:
                for node in path.nodes:
                    community_nodes.add(node.id)
            
            communities[seed] = {
                "seed_node": seed,
                "community_size": len(community_nodes),
                "members": list(community_nodes),
                "density": len(community_paths) / max(len(community_nodes), 1)
            }
        
        return communities

# Test functions
async def test_graph_traversal():
    """Test the graph traversal engine"""
    print("=== Testing Graph Traversal Engine ===")
    
    # Create traversal engine
    engine = GraphTraversalEngine()
    
    # Add nodes
    nodes = [
        GraphNode("alice", ["Person"], {"name": "Alice", "department": "Engineering"}),
        GraphNode("bob", ["Person"], {"name": "Bob", "department": "Engineering"}),
        GraphNode("charlie", ["Person"], {"name": "Charlie", "department": "Marketing"}),
        GraphNode("diana", ["Person"], {"name": "Diana", "department": "Sales"}),
        GraphNode("eve", ["Person"], {"name": "Eve", "department": "Engineering"})
    ]
    
    for node in nodes:
        engine.add_node(node)
    
    # Add edges
    edges = [
        GraphEdge("e1", "alice", "bob", "COLLABORATES_WITH", {"strength": 0.8}, 0.8),
        GraphEdge("e2", "bob", "charlie", "COLLABORATES_WITH", {"strength": 0.6}, 0.6),
        GraphEdge("e3", "charlie", "diana", "COLLABORATES_WITH", {"strength": 0.7}, 0.7),
        GraphEdge("e4", "alice", "eve", "COLLABORATES_WITH", {"strength": 0.9}, 0.9),
        GraphEdge("e5", "bob", "diana", "WORKS_ON", {"project": "ProjectX"}, 1.0)
    ]
    
    for edge in edges:
        engine.add_edge(edge)
    
    # Test different traversal strategies
    test_cases = [
        {
            "name": "Breadth-First Traversal",
            "strategy": TraversalStrategy.BREADTH_FIRST,
            "filters": QueryFilter(edge_types=["COLLABORATES_WITH"], max_depth=2, limit=10)
        },
        {
            "name": "Shortest Path",
            "strategy": TraversalStrategy.SHORTEST_PATH,
            "filters": QueryFilter(property_filters={"target_node": "diana"}, max_depth=5)
        },
        {
            "name": "Weighted Shortest Path",
            "strategy": TraversalStrategy.WEIGHTED_SHORTEST,
            "filters": QueryFilter(property_filters={"target_node": "diana"}, max_depth=5)
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        paths = await engine.traverse("alice", test_case["strategy"], test_case["filters"])
        
        for i, path in enumerate(paths[:3]):  # Show first 3 paths
            node_names = [node.properties.get("name", node.id) for node in path.nodes]
            edge_types = [edge.edge_type for edge in path.edges]
            print(f"  Path {i+1}: {' -> '.join(node_names)}")
            print(f"    Edge types: {edge_types}")
            print(f"    Total weight: {path.total_weight:.2f}")

async def test_query_optimizer():
    """Test the query optimizer"""
    print("\n=== Testing Query Optimizer ===")
    
    optimizer = QueryOptimizer()
    
    # Test different query plans
    query_plans = [
        {
            "strategy": TraversalStrategy.BREADTH_FIRST,
            "max_depth": 2,
            "filters": {"edge_types": ["COLLABORATES_WITH"]}
        },
        {
            "strategy": TraversalStrategy.WEIGHTED_SHORTEST,
            "max_depth": 5,
            "filters": {}
        },
        {
            "strategy": TraversalStrategy.DEPTH_FIRST,
            "max_depth": 8,
            "filters": {"property_filters": {"strength": 0.8}}
        }
    ]
    
    for i, plan in enumerate(query_plans):
        cost_analysis = await optimizer.analyze_query_cost(plan)
        print(f"\nQuery Plan {i+1}:")
        print(f"  Total Cost: {cost_analysis['total_cost']:.2f}")
        print(f"  Estimated Results: {cost_analysis['estimated_results']}")
        if cost_analysis['optimization_suggestions']:
            print(f"  Suggestions: {cost_analysis['optimization_suggestions']}")

async def test_advanced_patterns():
    """Test advanced query patterns"""
    print("\n=== Testing Advanced Query Patterns ===")
    
    # Create a more complex graph for testing
    engine = GraphTraversalEngine()
    
    # Add more nodes for complex patterns
    people = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace"]
    for person in people:
        engine.add_node(GraphNode(person, ["Person"], {"name": person.title()}))
    
    # Add collaboration network
    collaborations = [
        ("alice", "bob"), ("bob", "charlie"), ("charlie", "diana"),
        ("alice", "eve"), ("eve", "frank"), ("bob", "grace"),
        ("charlie", "frank"), ("diana", "grace"), ("alice", "charlie")
    ]
    
    for i, (from_person, to_person) in enumerate(collaborations):
        engine.add_edge(GraphEdge(f"collab_{i}", from_person, to_person, "COLLABORATES_WITH", 
                                {"strength": 0.7 + (i % 3) * 0.1}, 0.8))
    
    # Test advanced patterns
    patterns = AdvancedQueryPatterns(engine)
    
    # Test collaboration patterns
    collab_patterns = await patterns.find_collaboration_patterns("alice")
    print(f"Collaboration patterns for Alice:")
    print(f"  Direct collaborators: {collab_patterns['direct_collaborators']}")
    print(f"  Collaboration triangles: {collab_patterns['collaboration_triangles']}")
    print(f"  Network density: {collab_patterns['network_density']:.2f}")
    
    # Test community detection
    communities = await patterns.detect_communities(["alice", "diana"])
    print(f"\nCommunity detection:")
    for seed, community in communities.items():
        print(f"  {seed.title()}'s community: {community['community_size']} members")

async def main():
    """Run all traversal tests"""
    await test_graph_traversal()
    await test_query_optimizer()
    await test_advanced_patterns()

if __name__ == "__main__":
    asyncio.run(main())