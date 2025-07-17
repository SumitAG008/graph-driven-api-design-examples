#examples/chapter_06_implementation/microservices_patterns.py

from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import json
from abc import ABC, abstractmethod

class ServiceType(Enum):
    """Types of graph microservices"""
    NETWORK_ANALYSIS = "network_analysis"
    RECOMMENDATION = "recommendation"
    TRAVERSAL = "traversal"
    ANALYTICS = "analytics"
    SEARCH = "search"

@dataclass
class ServiceConfig:
    """Configuration for a graph microservice"""
    service_name: str
    service_type: ServiceType
    host: str = "localhost"
    port: int = 8000
    health_check_endpoint: str = "/health"
    dependencies: List[str] = field(default_factory=list)

@dataclass
class GraphEvent:
    """Event in the graph system"""
    event_id: str
    event_type: str
    source_service: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class GraphMicroservice(ABC):
    """Base class for graph microservices"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.health_status = "healthy"
        self.event_handlers: Dict[str, callable] = {}
        
    @abstractmethod
    async def start(self):
        """Start the microservice"""
        pass
        
    @abstractmethod
    async def stop(self):
        """Stop the microservice"""
        pass
        
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        pass
        
    def register_event_handler(self, event_type: str, handler: callable):
        """Register an event handler"""
        self.event_handlers[event_type] = handler
        
    async def handle_event(self, event: GraphEvent) -> Any:
        """Handle incoming events"""
        if event.event_type in self.event_handlers:
            return await self.event_handlers[event.event_type](event)
        else:
            print(f"No handler for event type: {event.event_type}")

class NetworkAnalysisService(GraphMicroservice):
    """Microservice focused on network analysis operations"""
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.graph_algorithms = {}
        self.metrics_cache = {}
        
    async def start(self):
        """Start the network analysis service"""
        print(f"Starting {self.config.service_name} on {self.config.host}:{self.config.port}")
        
        # Register event handlers
        self.register_event_handler("calculate_centrality", self._handle_centrality_calculation)
        self.register_event_handler("detect_communities", self._handle_community_detection)
        self.register_event_handler("analyze_influence", self._handle_influence_analysis)
        
        self.health_status = "healthy"
        
    async def stop(self):
        """Stop the network analysis service"""
        print(f"Stopping {self.config.service_name}")
        self.health_status = "stopped"
        
    async def health_check(self) -> Dict[str, Any]:
        """Health check for network analysis service"""
        return {
            "service": self.config.service_name,
            "status": self.health_status,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "cached_results": len(self.metrics_cache),
                "algorithms_loaded": len(self.graph_algorithms)
            }
        }
        
    async def calculate_centrality(self, graph_id: str, centrality_type: str,
                                 node_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Calculate centrality measures for nodes"""
        cache_key = f"{graph_id}:{centrality_type}:{hash(tuple(node_ids or []))}"
        
        # Check cache first
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
            
        # Simulate centrality calculation
        if centrality_type == "pagerank":
            result = await self._calculate_pagerank(graph_id, node_ids)
        elif centrality_type == "betweenness":
            result = await self._calculate_betweenness(graph_id, node_ids)
        elif centrality_type == "closeness":
            result = await self._calculate_closeness(graph_id, node_ids)
        else:
            raise ValueError(f"Unsupported centrality type: {centrality_type}")
            
        # Cache result
        self.metrics_cache[cache_key] = result
        return result
        
    async def detect_communities(self, graph_id: str, algorithm: str = "louvain",
                               resolution: float = 1.0) -> Dict[str, str]:
        """Detect communities in the network"""
        cache_key = f"{graph_id}:communities:{algorithm}:{resolution}"
        
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
            
        # Simulate community detection
        result = await self._detect_communities_impl(graph_id, algorithm, resolution)
        self.metrics_cache[cache_key] = result
        return result
        
    async def _handle_centrality_calculation(self, event: GraphEvent) -> Dict[str, float]:
        """Handle centrality calculation events"""
        payload = event.payload
        return await self.calculate_centrality(
            payload.get("graph_id"),
            payload.get("centrality_type"),
            payload.get("node_ids")
        )
        
    async def _handle_community_detection(self, event: GraphEvent) -> Dict[str, str]:
        """Handle community detection events"""
        payload = event.payload
        return await self.detect_communities(
            payload.get("graph_id"),
            payload.get("algorithm", "louvain"),
            payload.get("resolution", 1.0)
        )
        
    async def _handle_influence_analysis(self, event: GraphEvent) -> Dict[str, Any]:
        """Handle influence analysis events"""
        # Simulate influence analysis
        return {
            "influencers": ["alice", "bob", "charlie"],
            "influence_scores": {"alice": 0.85, "bob": 0.72, "charlie": 0.68},
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    async def _calculate_pagerank(self, graph_id: str, node_ids: Optional[List[str]]) -> Dict[str, float]:
        """Simulate PageRank calculation"""
        # In a real implementation, this would use actual graph algorithms
        nodes = node_ids or ["alice", "bob", "charlie", "diana", "eve"]
        return {node: 0.2 + (hash(node) % 100) / 1000 for node in nodes}
        
    async def _calculate_betweenness(self, graph_id: str, node_ids: Optional[List[str]]) -> Dict[str, float]:
        """Simulate betweenness centrality calculation"""
        nodes = node_ids or ["alice", "bob", "charlie", "diana", "eve"]
        return {node: 0.1 + (hash(node) % 80) / 1000 for node in nodes}
        
    async def _calculate_closeness(self, graph_id: str, node_ids: Optional[List[str]]) -> Dict[str, float]:
        """Simulate closeness centrality calculation"""
        nodes = node_ids or ["alice", "bob", "charlie", "diana", "eve"]
        return {node: 0.3 + (hash(node) % 70) / 1000 for node in nodes}
        
    async def _detect_communities_impl(self, graph_id: str, algorithm: str, resolution: float) -> Dict[str, str]:
        """Simulate community detection"""
        # Mock community assignment
        return {
            "alice": "community_1",
            "bob": "community_1", 
            "charlie": "community_2",
            "diana": "community_2",
            "eve": "community_3"
        }

class RecommendationService(GraphMicroservice):
    """Microservice focused on recommendation algorithms"""
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.recommendation_models = {}
        self.user_profiles = {}
        
    async def start(self):
        """Start the recommendation service"""
        print(f"Starting {self.config.service_name} on {self.config.host}:{self.config.port}")
        
        # Register event handlers
        self.register_event_handler("generate_recommendations", self._handle_recommendation_request)
        self.register_event_handler("update_user_profile", self._handle_profile_update)
        
        self.health_status = "healthy"
        
    async def stop(self):
        """Stop the recommendation service"""
        print(f"Stopping {self.config.service_name}")
        self.health_status = "stopped"
        
    async def health_check(self) -> Dict[str, Any]:
        """Health check for recommendation service"""
        return {
            "service": self.config.service_name,
            "status": self.health_status,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "active_models": len(self.recommendation_models),
                "user_profiles": len(self.user_profiles)
            }
        }
        
    async def generate_recommendations(self, user_id: str, item_type: str,
                                     algorithm: str = "collaborative", limit: int = 10) -> List[Dict[str, Any]]:
        """Generate recommendations for a user"""
        
        if algorithm == "collaborative":
            return await self._collaborative_filtering(user_id, item_type, limit)
        elif algorithm == "content":
            return await self._content_based_filtering(user_id, item_type, limit)
        elif algorithm == "hybrid":
            return await self._hybrid_recommendation(user_id, item_type, limit)
        else:
            raise ValueError(f"Unsupported recommendation algorithm: {algorithm}")
            
    async def _handle_recommendation_request(self, event: GraphEvent) -> List[Dict[str, Any]]:
        """Handle recommendation request events"""
        payload = event.payload
        return await self.generate_recommendations(
            payload.get("user_id"),
            payload.get("item_type"),
            payload.get("algorithm", "collaborative"),
            payload.get("limit", 10)
        )
        
    async def _handle_profile_update(self, event: GraphEvent) -> Dict[str, str]:
        """Handle user profile update events"""
        payload = event.payload
        user_id = payload.get("user_id")
        profile_data = payload.get("profile_data", {})
        
        self.user_profiles[user_id] = profile_data
        return {"status": "success", "user_id": user_id}
        
    async def _collaborative_filtering(self, user_id: str, item_type: str, limit: int) -> List[Dict[str, Any]]:
        """Collaborative filtering recommendations"""
        # Simulate collaborative filtering
        recommendations = [
            {
                "item_id": f"item_{i}",
                "item_type": item_type,
                "score": 0.9 - (i * 0.1),
                "reason": "Users with similar preferences also liked this",
                "algorithm": "collaborative_filtering"
            }
            for i in range(limit)
        ]
        return recommendations
        
    async def _content_based_filtering(self, user_id: str, item_type: str, limit: int) -> List[Dict[str, Any]]:
        """Content-based filtering recommendations"""
        # Simulate content-based filtering
        recommendations = [
            {
                "item_id": f"content_item_{i}",
                "item_type": item_type,
                "score": 0.85 - (i * 0.08),
                "reason": "Similar to items you've liked before",
                "algorithm": "content_based"
            }
            for i in range(limit)
        ]
        return recommendations
        
    async def _hybrid_recommendation(self, user_id: str, item_type: str, limit: int) -> List[Dict[str, Any]]:
        """Hybrid recommendation combining multiple algorithms"""
        # Get recommendations from both algorithms
        collab_recs = await self._collaborative_filtering(user_id, item_type, limit // 2)
        content_recs = await self._content_based_filtering(user_id, item_type, limit // 2)
        
        # Combine and re-score
        all_recs = collab_recs + content_recs
        for i, rec in enumerate(all_recs):
            rec["score"] = 0.95 - (i * 0.05)
            rec["algorithm"] = "hybrid"
            
        return all_recs[:limit]

class TraversalService(GraphMicroservice):
    """Microservice focused on graph traversal operations"""
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.path_cache = {}
        self.graph_structure = {}
        
    async def start(self):
        """Start the traversal service"""
        print(f"Starting {self.config.service_name} on {self.config.host}:{self.config.port}")
        
        # Register event handlers
        self.register_event_handler("find_shortest_path", self._handle_shortest_path)
        self.register_event_handler("extract_subgraph", self._handle_subgraph_extraction)
        
        self.health_status = "healthy"
        
    async def stop(self):
        """Stop the traversal service"""
        print(f"Stopping {self.config.service_name}")
        self.health_status = "stopped"
        
    async def health_check(self) -> Dict[str, Any]:
        """Health check for traversal service"""
        return {
            "service": self.config.service_name,
            "status": self.health_status,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "cached_paths": len(self.path_cache),
                "graph_nodes": len(self.graph_structure)
            }
        }
        
    async def find_shortest_path(self, start_id: str, end_id: str, max_depth: int = 6,
                               relationship_types: Optional[List[str]] = None) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two nodes"""
        cache_key = f"shortest:{start_id}:{end_id}:{max_depth}:{hash(tuple(relationship_types or []))}"
        
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
            
        # Simulate shortest path calculation
        path = await self._calculate_shortest_path(start_id, end_id, max_depth, relationship_types)
        
        if path:
            self.path_cache[cache_key] = path
            
        return path
        
    async def extract_subgraph(self, center_nodes: List[str], radius: int = 2,
                             filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract subgraph around center nodes"""
        
        subgraph = {
            "nodes": [],
            "edges": [],
            "metadata": {
                "center_nodes": center_nodes,
                "radius": radius,
                "filters": filters
            }
        }
        
        # Simulate subgraph extraction
        for i, center in enumerate(center_nodes):
            # Add center node
            subgraph["nodes"].append({
                "id": center,
                "labels": ["Person"],
                "properties": {"name": center.title(), "is_center": True}
            })
            
            # Add connected nodes within radius
            for depth in range(1, radius + 1):
                for j in range(2):  # Add 2 nodes per depth level
                    connected_id = f"{center}_connected_{depth}_{j}"
                    subgraph["nodes"].append({
                        "id": connected_id,
                        "labels": ["Person"],
                        "properties": {"name": connected_id.title(), "depth": depth}
                    })
                    
                    # Add edge
                    subgraph["edges"].append({
                        "id": f"edge_{center}_{connected_id}",
                        "from": center if depth == 1 else f"{center}_connected_{depth-1}_{j}",
                        "to": connected_id,
                        "type": "CONNECTED_TO",
                        "properties": {"weight": 1.0 / depth}
                    })
        
        return subgraph
        
    async def _handle_shortest_path(self, event: GraphEvent) -> Optional[List[Dict[str, Any]]]:
        """Handle shortest path request events"""
        payload = event.payload
        return await self.find_shortest_path(
            payload.get("start_id"),
            payload.get("end_id"),
            payload.get("max_depth", 6),
            payload.get("relationship_types")
        )
        
    async def _handle_subgraph_extraction(self, event: GraphEvent) -> Dict[str, Any]:
        """Handle subgraph extraction events"""
        payload = event.payload
        return await self.extract_subgraph(
            payload.get("center_nodes"),
            payload.get("radius", 2),
            payload.get("filters")
        )
        
    async def _calculate_shortest_path(self, start_id: str, end_id: str, max_depth: int,
                                     relationship_types: Optional[List[str]]) -> Optional[List[Dict[str, Any]]]:
        """Simulate shortest path calculation"""
        # Mock path calculation
        if start_id == end_id:
            return [{"id": start_id, "name": start_id.title()}]
            
        # Simulate a path
        path = [
            {"id": start_id, "name": start_id.title()},
            {"id": f"intermediate_1", "name": "Intermediate 1"},
            {"id": end_id, "name": end_id.title()}
        ]
        
        return path if len(path) <= max_depth else None

class EventDrivenGraphProcessor:
    """Process graph events in real-time with high throughput"""
    
    def __init__(self):
        self.services: Dict[str, GraphMicroservice] = {}
        self.event_queue = asyncio.Queue()
        self.running = False
        
    def register_service(self, service: GraphMicroservice):
        """Register a microservice"""
        self.services[service.config.service_name] = service
        
    async def start_all_services(self):
        """Start all registered services"""
        self.running = True
        
        # Start all services
        for service in self.services.values():
            await service.start()
            
        # Start event processor
        asyncio.create_task(self._process_events())
        
    async def stop_all_services(self):
        """Stop all services"""
        self.running = False
        
        for service in self.services.values():
            await service.stop()
            
    async def publish_event(self, event: GraphEvent):
        """Publish an event to the system"""
        await self.event_queue.put(event)
        
    async def _process_events(self):
        """Process events from the queue"""
        while self.running:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Route event to appropriate services
                await self._route_event(event)
                
            except asyncio.TimeoutError:
                continue  # Continue processing
            except Exception as e:
                print(f"Error processing event: {e}")
                
    async def _route_event(self, event: GraphEvent):
        """Route event to appropriate services based on event type"""
        
        # Define routing rules
        routing_rules = {
            "calculate_centrality": ["network_analysis_service"],
            "detect_communities": ["network_analysis_service"],
            "analyze_influence": ["network_analysis_service"],
            "generate_recommendations": ["recommendation_service"],
            "find_shortest_path": ["traversal_service"],
            "extract_subgraph": ["traversal_service"]
        }
        
        target_services = routing_rules.get(event.event_type, [])
        
        # Send event to target services
        for service_name in target_services:
            if service_name in self.services:
                try:
                    result = await self.services[service_name].handle_event(event)
                    print(f"Event {event.event_id} processed by {service_name}: {result is not None}")
                except Exception as e:
                    print(f"Error handling event {event.event_id} in {service_name}: {e}")

# Test functions
async def test_microservices_architecture():
    """Test the microservices architecture"""
    print("=== Testing Graph Microservices Architecture ===")
    
    # Create service configurations
    network_config = ServiceConfig("network_analysis_service", ServiceType.NETWORK_ANALYSIS, port=8001)
    recommendation_config = ServiceConfig("recommendation_service", ServiceType.RECOMMENDATION, port=8002)
    traversal_config = ServiceConfig("traversal_service", ServiceType.TRAVERSAL, port=8003)
    
    # Create services
    network_service = NetworkAnalysisService(network_config)
    recommendation_service = RecommendationService(recommendation_config)
    traversal_service = TraversalService(traversal_config)
    
    # Create event processor
    processor = EventDrivenGraphProcessor()
    processor.register_service(network_service)
    processor.register_service(recommendation_service)
    processor.register_service(traversal_service)
    
    # Start all services
    await processor.start_all_services()
    
    # Test events
    test_events = [
        GraphEvent("evt_1", "calculate_centrality", "client", {
            "graph_id": "company_graph",
            "centrality_type": "pagerank",
            "node_ids": ["alice", "bob", "charlie"]
        }),
        GraphEvent("evt_2", "generate_recommendations", "client", {
            "user_id": "alice",
            "item_type": "people",
            "algorithm": "collaborative",
            "limit": 5
        }),
        GraphEvent("evt_3", "find_shortest_path", "client", {
            "start_id": "alice",
            "end_id": "diana",
            "max_depth": 4
        })
    ]
    
    # Send events
    for event in test_events:
        await processor.publish_event(event)
        
    # Wait for processing
    await asyncio.sleep(2)
    
    # Check health of all services
    print("\nService Health Checks:")
    for service_name, service in processor.services.items():
        health = await service.health_check()
        print(f"{service_name}: {health['status']}")
        
    # Stop all services
    await processor.stop_all_services()

async def test_service_communication():
    """Test communication between services"""
    print("\n=== Testing Service Communication ===")
    
    # Create and start services
    network_service = NetworkAnalysisService(
        ServiceConfig("network_service", ServiceType.NETWORK_ANALYSIS)
    )
    await network_service.start()
    
    # Test direct service calls
    centrality_result = await network_service.calculate_centrality(
        "test_graph", "pagerank", ["alice", "bob", "charlie"]
    )
    print(f"Centrality calculation result: {centrality_result}")
    
    community_result = await network_service.detect_communities("test_graph", "louvain")
    print(f"Community detection result: {community_result}")
    
    await network_service.stop()

async def main():
    """Run all microservices tests"""
    await test_microservices_architecture()
    await test_service_communication()

if __name__ == "__main__":
    asyncio.run(main())