#examples/chapter_07_security/access_control.py

from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import json
from collections import defaultdict

class AccessLevel(Enum):
    """Access levels for graph resources"""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"

class RelationshipType(Enum):
    """Types of relationships that affect access control"""
    MANAGES = "MANAGES"
    REPORTS_TO = "REPORTS_TO"
    COLLABORATES_WITH = "COLLABORATES_WITH"
    SAME_TEAM = "SAME_TEAM"
    MENTORS = "MENTORS"
    FRIENDS_WITH = "FRIENDS_WITH"

@dataclass
class AccessContext:
    """Context information for access control decisions"""
    requester_id: str
    target_resource: str
    action: str
    timestamp: datetime = field(default_factory=datetime.now)
    purpose: Optional[str] = None
    location: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None

@dataclass
class AccessDecision:
    """Result of an access control evaluation"""
    granted: bool
    reasons: List[str] = field(default_factory=list)
    conditions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    valid_until: Optional[datetime] = None

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    from_entity: str
    to_entity: str
    relationship_type: RelationshipType
    strength: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    properties: Dict[str, Any] = field(default_factory=dict)

class RelationshipAnalyzer:
    """Analyzes relationships between entities for access control"""
    
    def __init__(self):
        self.relationships: List[Relationship] = []
        self.entity_cache: Dict[str, Dict[str, Any]] = {}
        
    def add_relationship(self, relationship: Relationship):
        """Add a relationship to the analyzer"""
        self.relationships.append(relationship)
        
    async def analyze_relationship(self, entity1: str, entity2: str) -> Dict[str, Any]:
        """Analyze the relationship between two entities"""
        direct_relationships = []
        indirect_paths = []
        
        # Find direct relationships
        for rel in self.relationships:
            if ((rel.from_entity == entity1 and rel.to_entity == entity2) or
                (rel.from_entity == entity2 and rel.to_entity == entity1)):
                direct_relationships.append(rel)
        
        # Find indirect paths (up to 3 degrees)
        indirect_paths = await self._find_indirect_paths(entity1, entity2, max_depth=3)
        
        # Calculate relationship strength
        overall_strength = self._calculate_overall_strength(direct_relationships, indirect_paths)
        
        return {
            "has_direct_relationship": len(direct_relationships) > 0,
            "direct_relationships": [rel.relationship_type.value for rel in direct_relationships],
            "indirect_paths": indirect_paths,
            "overall_strength": overall_strength,
            "collaboration_frequency": self._calculate_collaboration_frequency(entity1, entity2),
            "network_distance": len(indirect_paths[0]) if indirect_paths else float('inf')
        }
    
    async def _find_indirect_paths(self, start: str, end: str, max_depth: int) -> List[List[str]]:
        """Find indirect paths between entities using BFS"""
        if start == end:
            return [[start]]
            
        queue = [(start, [start])]
        visited = {start}
        paths = []
        
        for _ in range(max_depth):
            next_queue = []
            
            for current, path in queue:
                for rel in self.relationships:
                    next_entity = None
                    if rel.from_entity == current:
                        next_entity = rel.to_entity
                    elif rel.to_entity == current:
                        next_entity = rel.from_entity
                        
                    if next_entity and next_entity not in visited:
                        new_path = path + [next_entity]
                        if next_entity == end:
                            paths.append(new_path)
                        else:
                            next_queue.append((next_entity, new_path))
                            visited.add(next_entity)
            
            queue = next_queue
            if not queue or paths:  # Stop if we found paths or no more nodes to explore
                break
                
        return paths
    
    def _calculate_overall_strength(self, direct_rels: List[Relationship], 
                                   indirect_paths: List[List[str]]) -> float:
        """Calculate overall relationship strength"""
        if direct_rels:
            # Use maximum strength from direct relationships
            return max(rel.strength for rel in direct_rels)
        elif indirect_paths:
            # Decay strength based on path length
            shortest_path_length = min(len(path) for path in indirect_paths)
            return 1.0 / (shortest_path_length ** 2)
        else:
            return 0.0
    
    def _calculate_collaboration_frequency(self, entity1: str, entity2: str) -> float:
        """Calculate how frequently two entities collaborate"""
        collaboration_count = 0
        for rel in self.relationships:
            if (rel.relationship_type == RelationshipType.COLLABORATES_WITH and
                ((rel.from_entity == entity1 and rel.to_entity == entity2) or
                 (rel.from_entity == entity2 and rel.to_entity == entity1))):
                collaboration_count += 1
                
        # Normalize to a 0-1 scale (assuming max 10 collaborations is very high)
        return min(collaboration_count / 10.0, 1.0)

class GraphAccessController:
    """Multi-dimensional access control for graph systems"""
    
    def __init__(self):
        self.relationship_analyzer = RelationshipAnalyzer()
        self.access_policies: Dict[str, Any] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
    def add_relationship(self, relationship: Relationship):
        """Add a relationship for access control analysis"""
        self.relationship_analyzer.add_relationship(relationship)
        
    def set_access_policy(self, policy_name: str, policy: Dict[str, Any]):
        """Set an access control policy"""
        self.access_policies[policy_name] = policy
        
    async def evaluate_access(self, context: AccessContext) -> AccessDecision:
        """Evaluate access request considering multiple dimensions"""
        
        # Log the access attempt
        self._log_access_attempt(context)
        
        # Extract entity ID from resource
        target_entity_id = self._extract_entity_id(context.target_resource)
        
        # Check direct permissions
        direct_decision = await self._check_direct_permissions(
            context.requester_id, target_entity_id, context.action
        )
        
        # Check relationship-based permissions
        relationship_decision = await self._check_relationship_permissions(
            context.requester_id, target_entity_id, context.action, context
        )
        
        # Check context-based permissions
        context_decision = await self._check_context_permissions(
            context.requester_id, target_entity_id, context.action, context
        )
        
        # Combine decisions
        final_decision = self._combine_decisions([
            direct_decision,
            relationship_decision,
            context_decision
        ])
        
        # Log the final decision
        self._log_access_decision(context, final_decision)
        
        return final_decision
    
    async def _check_direct_permissions(self, requester_id: str, target_id: str, action: str) -> AccessDecision:
        """Check direct role-based permissions"""
        # Simplified role checking
        user_roles = self._get_user_roles(requester_id)
        
        if "admin" in user_roles:
            return AccessDecision(
                granted=True,
                reasons=["User has admin role"],
                confidence=1.0
            )
        elif "manager" in user_roles and action in ["read", "update"]:
            return AccessDecision(
                granted=True,
                reasons=["Manager role allows read/update access"],
                confidence=0.9
            )
        else:
            return AccessDecision(
                granted=False,
                reasons=["No direct permissions found"],
                confidence=0.8
            )
    
    async def _check_relationship_permissions(self, requester_id: str, target_id: str, 
                                           action: str, context: AccessContext) -> AccessDecision:
        """Check permissions based on relationships"""
        
        if requester_id == target_id:
            return AccessDecision(
                granted=True,
                reasons=["Self-access permitted"],
                confidence=1.0
            )
        
        # Analyze relationship
        relationship_analysis = await self.relationship_analyzer.analyze_relationship(
            requester_id, target_id
        )
        
        granted = False
        reasons = []
        confidence = 0.5
        
        # Manager relationship
        if RelationshipType.MANAGES.value in relationship_analysis["direct_relationships"]:
            granted = True
            reasons.append("Direct management relationship")
            confidence = 0.95
        
        # Collaboration relationship
        elif (RelationshipType.COLLABORATES_WITH.value in relationship_analysis["direct_relationships"] and
              relationship_analysis["collaboration_frequency"] > 0.7):
            if action == "read":
                granted = True
                reasons.append("Frequent collaboration partner")
                confidence = 0.8
        
        # Team member relationship
        elif RelationshipType.SAME_TEAM.value in relationship_analysis["direct_relationships"]:
            if action == "read":
                granted = True
                reasons.append("Same team member")
                confidence = 0.7
        
        # Network proximity
        elif relationship_analysis["network_distance"] <= 2 and action == "read":
            granted = True
            reasons.append(f"Close network connection (distance: {relationship_analysis['network_distance']})")
            confidence = 0.6
        
        return AccessDecision(
            granted=granted,
            reasons=reasons,
            confidence=confidence
        )
    
    async def _check_context_permissions(self, requester_id: str, target_id: str,
                                       action: str, context: AccessContext) -> AccessDecision:
        """Check permissions based on context"""
        
        granted = False
        reasons = []
        conditions = []
        confidence = 0.5
        
        # Emergency context
        if context.purpose == "emergency":
            granted = True
            reasons.append("Emergency access granted")
            conditions.append("Limited to emergency data only")
            confidence = 0.9
            # Set expiration for emergency access
            valid_until = datetime.now() + timedelta(hours=24)
        
        # Business hours context
        elif self._is_business_hours():
            confidence += 0.2
            reasons.append("Access during business hours")
        
        # Location-based context
        if context.location == "office":
            confidence += 0.1
            reasons.append("Access from office location")
        
        return AccessDecision(
            granted=granted,
            reasons=reasons,
            conditions=conditions,
            confidence=confidence,
            valid_until=locals().get('valid_until')
        )
    
    def _combine_decisions(self, decisions: List[AccessDecision]) -> AccessDecision:
        """Combine multiple access decisions using policy rules"""
        
        # If any decision explicitly grants access with high confidence, grant access
        for decision in decisions:
            if decision.granted and decision.confidence > 0.8:
                return AccessDecision(
                    granted=True,
                    reasons=decision.reasons,
                    conditions=decision.conditions,
                    confidence=decision.confidence,
                    valid_until=decision.valid_until
                )
        
        # If multiple decisions grant access, combine them
        granted_decisions = [d for d in decisions if d.granted]
        if len(granted_decisions) >= 2:
            combined_confidence = sum(d.confidence for d in granted_decisions) / len(granted_decisions)
            if combined_confidence > 0.6:
                all_reasons = []
                all_conditions = []
                for d in granted_decisions:
                    all_reasons.extend(d.reasons)
                    all_conditions.extend(d.conditions)
                
                return AccessDecision(
                    granted=True,
                    reasons=all_reasons,
                    conditions=all_conditions,
                    confidence=combined_confidence
                )
        
        # Default to deny
        return AccessDecision(
            granted=False,
            reasons=["Insufficient permissions"],
            confidence=0.9
        )
    
    def _extract_entity_id(self, resource: str) -> str:
        """Extract entity ID from resource string"""
        # Simple extraction - in practice this would be more sophisticated
        if "/" in resource:
            return resource.split("/")[-1]
        return resource
    
    def _get_user_roles(self, user_id: str) -> List[str]:
        """Get roles for a user - simplified implementation"""
        # In practice, this would query a role management system
        role_map = {
            "admin_001": ["admin"],
            "manager_001": ["manager"],
            "employee_001": ["employee"],
            "advait": ["employee"],
            "priya": ["manager"]
        }
        return role_map.get(user_id, ["employee"])
    
    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours"""
        now = datetime.now()
        return 9 <= now.hour <= 17 and now.weekday() < 5  # 9 AM to 5 PM, Monday to Friday
    
    def _log_access_attempt(self, context: AccessContext):
        """Log an access attempt"""
        self.audit_log.append({
            "timestamp": context.timestamp.isoformat(),
            "event_type": "access_attempt",
            "requester_id": context.requester_id,
            "target_resource": context.target_resource,
            "action": context.action,
            "purpose": context.purpose,
            "location": context.location
        })
    
    def _log_access_decision(self, context: AccessContext, decision: AccessDecision):
        """Log an access decision"""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": "access_decision",
            "requester_id": context.requester_id,
            "target_resource": context.target_resource,
            "action": context.action,
            "granted": decision.granted,
            "reasons": decision.reasons,
            "confidence": decision.confidence
        })

class GraphAuditLogger:
    """Comprehensive audit logging for graph operations"""
    
    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []
        
    async def log_graph_access(self, access_event: Dict[str, Any]) -> Dict[str, Any]:
        """Log graph access with full context"""
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "graph_access",
            "user_id": access_event.get("user_id"),
            "resource_type": access_event.get("resource_type"),
            "resource_id": access_event.get("resource_id"),
            "action": access_event.get("action"),
            "outcome": access_event.get("outcome"),
            "graph_context": {
                "traversal_path": access_event.get("traversal_path", []),
                "relationship_types_accessed": access_event.get("relationship_types", []),
                "network_distance": access_event.get("network_distance"),
                "query_complexity": access_event.get("query_complexity", 1),
                "data_volume_accessed": access_event.get("data_volume", 0)
            },
            "access_decision": {
                "decision_factors": access_event.get("decision_factors", []),
                "permissions_used": access_event.get("permissions_used", []),
                "policy_rules_applied": access_event.get("policy_rules", [])
            },
            "privacy_context": {
                "privacy_level": access_event.get("privacy_level", "standard"),
                "anonymization_applied": access_event.get("anonymization", False),
                "differential_privacy_budget": access_event.get("privacy_budget", 0)
            }
        }
        
        self.audit_log.append(audit_entry)
        return audit_entry
    
    async def generate_compliance_report(self, report_type: str, 
                                       start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance reports for various regulations"""
        
        # Filter logs by date range
        filtered_logs = [
            log for log in self.audit_log
            if start_date <= datetime.fromisoformat(log["timestamp"]) <= end_date
        ]
        
        if report_type == "GDPR":
            return await self._generate_gdpr_report(filtered_logs)
        elif report_type == "access_summary":
            return await self._generate_access_summary_report(filtered_logs)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
    
    async def _generate_gdpr_report(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate GDPR compliance report"""
        
        data_subject_accesses = [log for log in logs if log.get("action") == "data_access_request"]
        personal_data_accesses = [log for log in logs if "personal" in log.get("privacy_context", {}).get("privacy_level", "")]
        
        return {
            "report_type": "GDPR",
            "period": {
                "start": logs[0]["timestamp"] if logs else None,
                "end": logs[-1]["timestamp"] if logs else None
            },
            "summary": {
                "total_data_accesses": len([log for log in logs if log.get("action") in ["read", "query"]]),
                "data_subject_requests": len(data_subject_accesses),
                "personal_data_accesses": len(personal_data_accesses),
                "consent_based_accesses": len([log for log in logs if "consent" in log.get("access_decision", {}).get("decision_factors", [])]),
            },
            "compliance_indicators": {
                "purpose_limitation": self._check_purpose_limitation(logs),
                "data_minimization": self._check_data_minimization(logs),
                "consent_management": self._check_consent_management(logs)
            }
        }
    
    async def _generate_access_summary_report(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate access summary report"""
        
        users = set(log.get("user_id") for log in logs if log.get("user_id"))
        successful_accesses = [log for log in logs if log.get("outcome") == "success"]
        failed_accesses = [log for log in logs if log.get("outcome") == "denied"]
        
        return {
            "report_type": "access_summary",
            "summary": {
                "total_access_attempts": len(logs),
                "unique_users": len(users),
                "successful_accesses": len(successful_accesses),
                "failed_accesses": len(failed_accesses),
                "success_rate": len(successful_accesses) / len(logs) if logs else 0
            },
            "top_accessed_resources": self._get_top_accessed_resources(logs),
            "access_patterns": self._analyze_access_patterns(logs)
        }
    
    def _check_purpose_limitation(self, logs: List[Dict[str, Any]]) -> float:
        """Check compliance with purpose limitation principle"""
        # Simplified check - in practice this would be more sophisticated
        purpose_compliant = sum(1 for log in logs if log.get("purpose") is not None)
        return purpose_compliant / len(logs) if logs else 1.0
    
    def _check_data_minimization(self, logs: List[Dict[str, Any]]) -> float:
        """Check compliance with data minimization principle"""
        # Check if data volume accessed is reasonable
        reasonable_accesses = sum(1 for log in logs 
                                if log.get("graph_context", {}).get("data_volume_accessed", 0) <= 100)
        return reasonable_accesses / len(logs) if logs else 1.0
    
    def _check_consent_management(self, logs: List[Dict[str, Any]]) -> float:
        """Check consent management compliance"""
        consent_based = sum(1 for log in logs 
                          if "consent" in log.get("access_decision", {}).get("decision_factors", []))
        return consent_based / len(logs) if logs else 0.0
    
    def _get_top_accessed_resources(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get most frequently accessed resources"""
        resource_counts = defaultdict(int)
        for log in logs:
            resource = log.get("resource_id")
            if resource:
                resource_counts[resource] += 1
        
        top_resources = sorted(resource_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        return [{"resource_id": resource, "access_count": count} for resource, count in top_resources]
    
    def _analyze_access_patterns(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze access patterns"""
        hourly_distribution = defaultdict(int)
        for log in logs:
            hour = datetime.fromisoformat(log["timestamp"]).hour
            hourly_distribution[hour] += 1
        
        return {
            "peak_access_hour": max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None,
            "hourly_distribution": dict(hourly_distribution)
        }

# Test functions
async def test_access_control():
    """Test the access control system"""
    print("=== Testing Graph Access Control ===")
    
    # Create access controller
    controller = GraphAccessController()
    
    # Add some relationships
    controller.add_relationship(Relationship("advait", "priya", RelationshipType.REPORTS_TO, 1.0))
    controller.add_relationship(Relationship("advait", "bob", RelationshipType.COLLABORATES_WITH, 0.8))
    controller.add_relationship(Relationship("priya", "charlie", RelationshipType.MANAGES, 1.0))
    controller.add_relationship(Relationship("advait", "bob", RelationshipType.SAME_TEAM, 1.0))
    
    # Test access scenarios
    test_cases = [
        {
            "name": "Self access",
            "context": AccessContext("advait", "/users/advait", "read")
        },
        {
            "name": "Manager access to report",
            "context": AccessContext("priya", "/users/advait", "read")
        },
        {
            "name": "Peer access",
            "context": AccessContext("bob", "/users/advait", "read")
        },
        {
            "name": "Unauthorized access",
            "context": AccessContext("stranger", "/users/advait", "read")
        },
        {
            "name": "Emergency access",
            "context": AccessContext("doctor", "/users/patient", "read", purpose="emergency")
        }
    ]
    
    for test_case in test_cases:
        decision = await controller.evaluate_access(test_case["context"])
        print(f"\n{test_case['name']}:")
        print(f"  Granted: {decision.granted}")
        print(f"  Reasons: {decision.reasons}")
        print(f"  Confidence: {decision.confidence:.2f}")
        if decision.conditions:
            print(f"  Conditions: {decision.conditions}")

async def test_audit_logging():
    """Test the audit logging system"""
    print("\n=== Testing Audit Logging ===")
    
    logger = GraphAuditLogger()
    
    # Log some access events
    events = [
        {
            "user_id": "advait",
            "resource_type": "user_profile",
            "resource_id": "priya",
            "action": "read",
            "outcome": "success",
            "decision_factors": ["direct_relationship", "same_team"],
            "privacy_level": "standard"
        },
        {
            "user_id": "bob",
            "resource_type": "user_profile", 
            "resource_id": "charlie",
            "action": "read",
            "outcome": "denied",
            "decision_factors": ["no_relationship"],
            "privacy_level": "restricted"
        }
    ]
    
    for event in events:
        await logger.log_graph_access(event)
    
    # Generate compliance report
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    
    report = await logger.generate_compliance_report("access_summary", start_date, end_date)
    print(f"Access Summary Report:")
    print(json.dumps(report, indent=2))

async def main():
    """Run all security tests"""
    await test_access_control()
    await test_audit_logging()

if __name__ == "__main__":
    asyncio.run(main())