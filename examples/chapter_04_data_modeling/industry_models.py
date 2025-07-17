#examples/chapter_04_data_modeling/industry_models.py

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import asyncio
import json

class EntityType(Enum):
    """Common entity types across domains"""
    PERSON = "Person"
    PRODUCT = "Product"
    CATEGORY = "Category"
    BRAND = "Brand"
    CUSTOMER = "Customer"
    ORDER = "Order"
    ACCOUNT = "Account"
    TRANSACTION = "Transaction"
    PATIENT = "Patient"
    PROVIDER = "Provider"
    CONDITION = "Condition"
    MEDICATION = "Medication"

@dataclass
class GraphEntity:
    """Base graph entity"""
    id: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class GraphRelationship:
    """Base graph relationship"""
    id: str
    relationship_type: str
    from_entity: str
    to_entity: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class EcommerceGraphModel:
    """Comprehensive graph model for e-commerce systems"""
    
    def __init__(self):
        self.entities: Dict[str, GraphEntity] = {}
        self.relationships: Dict[str, GraphRelationship] = {}
        
    def create_customer(self, customer_id: str, name: str, email: str, 
                       tier: str = "Standard") -> GraphEntity:
        """Create a customer entity"""
        customer = GraphEntity(
            id=customer_id,
            entity_type=EntityType.CUSTOMER,
            properties={
                "name": name,
                "email": email,
                "tier": tier,
                "registration_date": date.today().isoformat()
            }
        )
        self.entities[customer_id] = customer
        return customer
    
    def create_product(self, product_id: str, name: str, category: str, 
                      brand: str, price: float) -> GraphEntity:
        """Create a product entity"""
        product = GraphEntity(
            id=product_id,
            entity_type=EntityType.PRODUCT,
            properties={
                "name": name,
                "category": category,
                "brand": brand,
                "price": price,
                "availability": "in_stock"
            }
        )
        self.entities[product_id] = product
        return product
    
    def create_purchase_relationship(self, customer_id: str, product_id: str,
                                   quantity: int, price: float, 
                                   satisfaction: int = None) -> GraphRelationship:
        """Create a purchase relationship"""
        rel_id = f"purchase_{customer_id}_{product_id}_{int(datetime.now().timestamp())}"
        relationship = GraphRelationship(
            id=rel_id,
            relationship_type="PURCHASED",
            from_entity=customer_id,
            to_entity=product_id,
            properties={
                "quantity": quantity,
                "price_paid": price,
                "satisfaction": satisfaction,
                "purchase_date": datetime.now().isoformat()
            }
        )
        self.relationships[rel_id] = relationship
        return relationship
    
    def create_recommendation_relationship(self, product1_id: str, product2_id: str,
                                         similarity_score: float) -> GraphRelationship:
        """Create a product recommendation relationship"""
        rel_id = f"recommends_{product1_id}_{product2_id}"
        relationship = GraphRelationship(
            id=rel_id,
            relationship_type="RECOMMENDS",
            from_entity=product1_id,
            to_entity=product2_id,
            properties={
                "similarity_score": similarity_score,
                "algorithm": "collaborative_filtering"
            }
        )
        self.relationships[rel_id] = relationship
        return relationship
    
    async def generate_recommendations(self, customer_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Generate product recommendations for a customer"""
        if customer_id not in self.entities:
            return []
            
        # Find products purchased by this customer
        purchased_products = set()
        for rel in self.relationships.values():
            if (rel.relationship_type == "PURCHASED" and 
                rel.from_entity == customer_id):
                purchased_products.add(rel.to_entity)
        
        # Find recommended products
        recommendations = []
        for rel in self.relationships.values():
            if (rel.relationship_type == "RECOMMENDS" and 
                rel.from_entity in purchased_products and
                rel.to_entity not in purchased_products):
                
                product = self.entities.get(rel.to_entity)
                if product:
                    recommendations.append({
                        "product_id": product.id,
                        "product_name": product.properties.get("name"),
                        "similarity_score": rel.properties.get("similarity_score"),
                        "reason": f"Customers who bought {self.entities[rel.from_entity].properties.get('name')} also liked this"
                    })
        
        # Sort by similarity score and return top recommendations
        recommendations.sort(key=lambda x: x["similarity_score"], reverse=True)
        return recommendations[:limit]

class FinancialNetworkModel:
    """Comprehensive financial network model for risk analysis"""
    
    def __init__(self):
        self.entities: Dict[str, GraphEntity] = {}
        self.relationships: Dict[str, GraphRelationship] = {}
        
    def create_account(self, account_id: str, account_type: str, 
                      owner_id: str, balance: float) -> GraphEntity:
        """Create a financial account"""
        account = GraphEntity(
            id=account_id,
            entity_type=EntityType.ACCOUNT,
            properties={
                "account_type": account_type,
                "owner_id": owner_id,
                "balance": balance,
                "status": "active",
                "opened_date": date.today().isoformat()
            }
        )
        self.entities[account_id] = account
        return account
    
    def create_transaction(self, from_account: str, to_account: str,
                          amount: float, transaction_type: str) -> GraphRelationship:
        """Create a transaction relationship"""
        rel_id = f"txn_{int(datetime.now().timestamp())}"
        relationship = GraphRelationship(
            id=rel_id,
            relationship_type="TRANSACTION",
            from_entity=from_account,
            to_entity=to_account,
            properties={
                "amount": amount,
                "transaction_type": transaction_type,
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
        )
        self.relationships[rel_id] = relationship
        return relationship
    
    async def detect_suspicious_patterns(self) -> List[Dict[str, Any]]:
        """Detect potentially suspicious transaction patterns"""
        suspicious_patterns = []
        
        # Pattern 1: Rapid multiple transactions
        account_transactions = defaultdict(list)
        for rel in self.relationships.values():
            if rel.relationship_type == "TRANSACTION":
                account_transactions[rel.from_entity].append(rel)
        
        for account_id, transactions in account_transactions.items():
            if len(transactions) > 10:  # More than 10 transactions
                total_amount = sum(t.properties.get("amount", 0) for t in transactions)
                suspicious_patterns.append({
                    "pattern_type": "high_frequency_transactions",
                    "account_id": account_id,
                    "transaction_count": len(transactions),
                    "total_amount": total_amount,
                    "risk_score": min(len(transactions) / 10.0, 1.0)
                })
        
        return suspicious_patterns

class HealthcareKnowledgeGraph:
    """Comprehensive knowledge graph for healthcare and life sciences"""
    
    def __init__(self):
        self.entities: Dict[str, GraphEntity] = {}
        self.relationships: Dict[str, GraphRelationship] = {}
        
    def create_patient(self, patient_id: str, age: int, gender: str) -> GraphEntity:
        """Create a patient entity"""
        patient = GraphEntity(
            id=patient_id,
            entity_type=EntityType.PATIENT,
            properties={
                "age": age,
                "gender": gender,
                "registration_date": date.today().isoformat()
            }
        )
        self.entities[patient_id] = patient
        return patient
    
    def create_condition(self, condition_id: str, name: str, 
                        category: str, severity: str) -> GraphEntity:
        """Create a medical condition entity"""
        condition = GraphEntity(
            id=condition_id,
            entity_type=EntityType.CONDITION,
            properties={
                "name": name,
                "category": category,
                "severity": severity,
                "icd_code": f"ICD-{condition_id}"
            }
        )
        self.entities[condition_id] = condition
        return condition
    
    def create_diagnosis_relationship(self, patient_id: str, condition_id: str,
                                    provider_id: str, confidence: float) -> GraphRelationship:
        """Create a diagnosis relationship"""
        rel_id = f"diagnosis_{patient_id}_{condition_id}"
        relationship = GraphRelationship(
            id=rel_id,
            relationship_type="DIAGNOSED_WITH",
            from_entity=patient_id,
            to_entity=condition_id,
            properties={
                "provider_id": provider_id,
                "confidence": confidence,
                "diagnosis_date": date.today().isoformat(),
                "status": "active"
            }
        )
        self.relationships[rel_id] = relationship
        return relationship
    
    async def find_similar_patients(self, patient_id: str) -> List[Dict[str, Any]]:
        """Find patients with similar conditions"""
        if patient_id not in self.entities:
            return []
            
        # Get conditions for the target patient
        patient_conditions = set()
        for rel in self.relationships.values():
            if (rel.relationship_type == "DIAGNOSED_WITH" and 
                rel.from_entity == patient_id):
                patient_conditions.add(rel.to_entity)
        
        # Find similar patients
        similar_patients = []
        patient_similarities = defaultdict(int)
        
        for rel in self.relationships.values():
            if (rel.relationship_type == "DIAGNOSED_WITH" and 
                rel.from_entity != patient_id and
                rel.to_entity in patient_conditions):
                patient_similarities[rel.from_entity] += 1
        
        for similar_patient_id, shared_conditions in patient_similarities.items():
            similarity_score = shared_conditions / len(patient_conditions)
            if similarity_score > 0.5:  # At least 50% similar
                similar_patients.append({
                    "patient_id": similar_patient_id,
                    "shared_conditions": shared_conditions,
                    "similarity_score": similarity_score
                })
        
        similar_patients.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similar_patients

class SocialNetworkModel:
    """Multi-layer social network model"""
    
    def __init__(self):
        self.entities: Dict[str, GraphEntity] = {}
        self.relationships: Dict[str, GraphRelationship] = {}
        
    def create_user(self, user_id: str, name: str, age: int, location: str) -> GraphEntity:
        """Create a user entity"""
        user = GraphEntity(
            id=user_id,
            entity_type=EntityType.PERSON,
            properties={
                "name": name,
                "age": age,
                "location": location,
                "join_date": date.today().isoformat()
            }
        )
        self.entities[user_id] = user
        return user
    
    def create_friendship(self, user1_id: str, user2_id: str, 
                         relationship_strength: float = 1.0) -> GraphRelationship:
        """Create a friendship relationship"""
        rel_id = f"friendship_{user1_id}_{user2_id}"
        relationship = GraphRelationship(
            id=rel_id,
            relationship_type="FRIENDS_WITH",
            from_entity=user1_id,
            to_entity=user2_id,
            properties={
                "strength": relationship_strength,
                "since": date.today().isoformat(),
                "bidirectional": True
            }
        )
        self.relationships[rel_id] = relationship
        return relationship
    
    async def detect_communities(self) -> Dict[str, List[str]]:
        """Simple community detection based on friendship connections"""
        from collections import defaultdict, deque
        
        # Build adjacency list
        graph = defaultdict(set)
        for rel in self.relationships.values():
            if rel.relationship_type == "FRIENDS_WITH":
                graph[rel.from_entity].add(rel.to_entity)
                graph[rel.to_entity].add(rel.from_entity)  # bidirectional
        
        visited = set()
        communities = {}
        community_id = 0
        
        for user_id in self.entities:
            if user_id not in visited:
                # BFS to find connected component
                community = []
                queue = deque([user_id])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        community.append(current)
                        
                        for neighbor in graph[current]:
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                if len(community) > 1:  # Only include communities with multiple members
                    communities[f"community_{community_id}"] = community
                    community_id += 1
        
        return communities

# Testing functions
async def test_ecommerce_model():
    """Test the e-commerce graph model"""
    print("=== Testing E-commerce Model ===")
    
    model = EcommerceGraphModel()
    
    # Create customers
    customer1 = model.create_customer("cust_001", "Advait Sharma", "advait@example.com", "Premium")
    customer2 = model.create_customer("cust_002", "Priya Patel", "priya@example.com", "Standard")
    
    # Create products
    laptop = model.create_product("prod_001", "Gaming Laptop", "Electronics", "TechBrand", 75000)
    mouse = model.create_product("prod_002", "Wireless Mouse", "Electronics", "TechBrand", 2500)
    keyboard = model.create_product("prod_003", "Mechanical Keyboard", "Electronics", "TechBrand", 5000)
    
    # Create purchases
    model.create_purchase_relationship("cust_001", "prod_001", 1, 75000, 9)
    model.create_purchase_relationship("cust_002", "prod_001", 1, 75000, 8)
    model.create_purchase_relationship("cust_001", "prod_002", 1, 2500, 10)
    
    # Create recommendations
    model.create_recommendation_relationship("prod_001", "prod_002", 0.85)
    model.create_recommendation_relationship("prod_001", "prod_003", 0.75)
    
    # Generate recommendations
    recommendations = await model.generate_recommendations("cust_002")
    print(f"Recommendations for customer 2: {json.dumps(recommendations, indent=2)}")

async def test_healthcare_model():
    """Test the healthcare knowledge graph"""
    print("\n=== Testing Healthcare Model ===")
    
    model = HealthcareKnowledgeGraph()
    
    # Create patients
    patient1 = model.create_patient("pat_001", 45, "M")
    patient2 = model.create_patient("pat_002", 47, "M")
    patient3 = model.create_patient("pat_003", 35, "F")
    
    # Create conditions
    hypertension = model.create_condition("cond_001", "Hypertension", "Cardiovascular", "Moderate")
    diabetes = model.create_condition("cond_002", "Type 2 Diabetes", "Endocrine", "Moderate")
    obesity = model.create_condition("cond_003", "Obesity", "Metabolic", "Mild")
    
    # Create diagnoses
    model.create_diagnosis_relationship("pat_001", "cond_001", "prov_001", 0.95)
    model.create_diagnosis_relationship("pat_001", "cond_002", "prov_001", 0.90)
    model.create_diagnosis_relationship("pat_002", "cond_001", "prov_002", 0.88)
    model.create_diagnosis_relationship("pat_002", "cond_003", "prov_002", 0.85)
    
    # Find similar patients
    similar = await model.find_similar_patients("pat_001")
    print(f"Patients similar to pat_001: {json.dumps(similar, indent=2)}")

async def main():
    """Run all model tests"""
    await test_ecommerce_model()
    await test_healthcare_model()

if __name__ == "__main__":
    # Add missing import
    from collections import defaultdict
    asyncio.run(main())