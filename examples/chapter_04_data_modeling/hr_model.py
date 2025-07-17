# examples/chapter_04_data_modeling/hr_model.py

from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import asyncio
import json
from abc import ABC, abstractmethod

class EmploymentType(Enum):
    """Employment types"""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERN = "intern"

class PerformanceLevel(Enum):
    """Performance rating levels"""
    OUTSTANDING = "outstanding"
    EXCEEDS = "exceeds"
    MEETS = "meets"
    BELOW = "below"
    UNSATISFACTORY = "unsatisfactory"

@dataclass
class Employee:
    """Employee entity with comprehensive attributes"""
    id: str
    name: str
    title: str
    department: str
    hire_date: str
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    performance_rating: float = 3.0
    salary_grade: str = "L1"
    tenure_months: int = 0
    flight_risk_score: float = 0.0
    email: Optional[str] = None
    manager_id: Optional[str] = None
    location: str = "Remote"
    skills: List[str] = field(default_factory=list)

@dataclass
class Position:
    """Position/Job role entity"""
    id: str
    title: str
    department: str
    required_skills: List[str]
    min_experience: int
    salary_range: str
    level: str = "Mid"
    status: str = "Open"

@dataclass
class Skill:
    """Skill entity"""
    id: str
    name: str
    category: str
    complexity_level: str = "Intermediate"
    market_demand: str = "Medium"

@dataclass
class Project:
    """Project entity"""
    id: str
    name: str
    status: str
    duration_months: int
    team_size: int
    priority: str = "Medium"
    budget: Optional[float] = None

@dataclass
class Relationship:
    """Base relationship class"""
    from_id: str
    to_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class HRGraphModel:
    """HR Domain Graph Model for testing"""
    
    def __init__(self):
        self.employees: Dict[str, Employee] = {}
        self.positions: Dict[str, Position] = {}
        self.departments: Dict[str, Dict[str, Any]] = {}
        self.skills: Dict[str, Skill] = {}
        self.projects: Dict[str, Project] = {}
        self.relationships: List[Relationship] = []
        self._initialize_hr_data()
    
    def _initialize_hr_data(self):
        """Initialize sample HR data for testing"""
        
        # Sample employees
        self.employees = {
            "emp_001": Employee(
                id="emp_001",
                name="Advait Sharma",
                title="Senior Developer",
                department="Engineering",
                hire_date="2022-03-15",
                performance_rating=4.2,
                salary_grade="L4",
                tenure_months=24,
                flight_risk_score=0.2,
                email="advait@company.com",
                manager_id="emp_002",
                location="Mumbai"
            ),
            "emp_002": Employee(
                id="emp_002",
                name="Priya Patel",
                title="Tech Lead",
                department="Engineering",
                hire_date="2021-01-10",
                performance_rating=4.5,
                salary_grade="L5",
                tenure_months=36,
                flight_risk_score=0.1,
                email="priya@company.com",
                manager_id="emp_005",
                location="Bangalore"
            ),
            "emp_003": Employee(
                id="emp_003",
                name="Kavya Singh",
                title="Product Manager",
                department="Product",
                hire_date="2020-06-01",
                performance_rating=4.3,
                salary_grade="L5",
                tenure_months=48,
                flight_risk_score=0.3,
                email="kavya@company.com",
                location="Delhi"
            ),
            "emp_004": Employee(
                id="emp_004",
                name="Rahul Kumar",
                title="Junior Developer",
                department="Engineering",
                hire_date="2023-08-01",
                performance_rating=3.8,
                salary_grade="L3",
                tenure_months=6,
                flight_risk_score=0.7,
                email="rahul@company.com",
                manager_id="emp_002",
                location="Pune"
            ),
            "emp_005": Employee(
                id="emp_005",
                name="Ananya Gupta",
                title="Engineering Manager",
                department="Engineering",
                hire_date="2019-03-15",
                performance_rating=4.6,
                salary_grade="L6",
                tenure_months=60,
                flight_risk_score=0.1,
                email="ananya@company.com",
                location="Mumbai"
            )
        }
        
        # Sample positions
        self.positions = {
            "pos_001": Position(
                id="pos_001",
                title="Senior Developer",
                department="Engineering",
                required_skills=["Python", "JavaScript", "System Design"],
                min_experience=4,
                salary_range="70-90k",
                level="Senior"
            ),
            "pos_002": Position(
                id="pos_002",
                title="Tech Lead",
                department="Engineering",
                required_skills=["Python", "JavaScript", "Leadership", "System Design"],
                min_experience=6,
                salary_range="90-120k",
                level="Lead"
            ),
            "pos_003": Position(
                id="pos_003",
                title="Engineering Manager",
                department="Engineering",
                required_skills=["Leadership", "Team Management", "Technical Strategy"],
                min_experience=8,
                salary_range="120-150k",
                level="Manager"
            )
        }
        
        # Sample skills
        self.skills = {
            "skill_001": Skill(id="skill_001", name="Python", category="Programming"),
            "skill_002": Skill(id="skill_002", name="JavaScript", category="Programming"),
            "skill_003": Skill(id="skill_003", name="Leadership", category="Soft Skills"),
            "skill_004": Skill(id="skill_004", name="System Design", category="Architecture"),
            "skill_005": Skill(id="skill_005", name="Product Management", category="Business"),
            "skill_006": Skill(id="skill_006", name="Team Management", category="Leadership")
        }
        
        # Sample projects
        self.projects = {
            "proj_001": Project(
                id="proj_001",
                name="Customer Portal Redesign",
                status="Active",
                duration_months=6,
                team_size=5,
                priority="High"
            ),
            "proj_002": Project(
                id="proj_002",
                name="API Gateway Migration",
                status="Completed",
                duration_months=4,
                team_size=3,
                priority="Medium"
            )
        }
        
        # Sample departments
        self.departments = {
            "Engineering": {
                "id": "dept_001",
                "name": "Engineering",
                "head": "emp_005",
                "budget": 5000000,
                "employee_count": 4
            },
            "Product": {
                "id": "dept_002", 
                "name": "Product",
                "head": "emp_003",
                "budget": 2000000,
                "employee_count": 1
            }
        }
        
        # Sample relationships
        self.relationships = [
            # Reporting relationships
            Relationship("emp_001", "emp_002", "REPORTS_TO", 
                        {"since": "2022-03-15", "relationship_strength": 0.9}),
            Relationship("emp_002", "emp_005", "REPORTS_TO", 
                        {"since": "2021-01-10", "relationship_strength": 0.8}),
            Relationship("emp_004", "emp_002", "REPORTS_TO", 
                        {"since": "2023-08-01", "relationship_strength": 0.7}),
            
            # Skill relationships
            Relationship("emp_001", "skill_001", "HAS_SKILL", 
                        {"proficiency": 4.5, "years_experience": 5, "certified": True}),
            Relationship("emp_001", "skill_002", "HAS_SKILL", 
                        {"proficiency": 4.0, "years_experience": 4, "certified": False}),
            Relationship("emp_001", "skill_004", "HAS_SKILL", 
                        {"proficiency": 3.5, "years_experience": 2, "certified": False}),
            Relationship("emp_002", "skill_001", "HAS_SKILL", 
                        {"proficiency": 4.8, "years_experience": 7, "certified": True}),
            Relationship("emp_002", "skill_003", "HAS_SKILL", 
                        {"proficiency": 4.2, "years_experience": 3, "certified": False}),
            Relationship("emp_002", "skill_004", "HAS_SKILL", 
                        {"proficiency": 4.5, "years_experience": 4, "certified": True}),
            Relationship("emp_003", "skill_005", "HAS_SKILL", 
                        {"proficiency": 4.7, "years_experience": 6, "certified": True}),
            Relationship("emp_004", "skill_001", "HAS_SKILL", 
                        {"proficiency": 3.0, "years_experience": 1, "certified": False}),
            Relationship("emp_004", "skill_002", "HAS_SKILL", 
                        {"proficiency": 3.5, "years_experience": 1, "certified": False}),
            Relationship("emp_005", "skill_003", "HAS_SKILL", 
                        {"proficiency": 4.8, "years_experience": 8, "certified": True}),
            Relationship("emp_005", "skill_006", "HAS_SKILL", 
                        {"proficiency": 4.6, "years_experience": 5, "certified": True}),
            
            # Project participation
            Relationship("emp_001", "proj_001", "WORKS_ON", 
                        {"role": "Lead Developer", "allocation": 0.8, "start_date": "2024-01-01"}),
            Relationship("emp_002", "proj_001", "WORKS_ON", 
                        {"role": "Tech Lead", "allocation": 0.6, "start_date": "2024-01-01"}),
            Relationship("emp_004", "proj_001", "WORKS_ON", 
                        {"role": "Developer", "allocation": 1.0, "start_date": "2024-01-15"}),
            Relationship("emp_001", "proj_002", "WORKED_ON", 
                        {"role": "Developer", "contribution": "High", "end_date": "2023-12-31"}),
            Relationship("emp_002", "proj_002", "WORKED_ON", 
                        {"role": "Tech Lead", "contribution": "High", "end_date": "2023-12-31"}),
            
            # Mentorship relationships
            Relationship("emp_002", "emp_004", "MENTORS", 
                        {"since": "2023-08-01", "focus": "Technical Skills", "meeting_frequency": "Weekly"}),
            Relationship("emp_005", "emp_002", "MENTORS", 
                        {"since": "2021-01-10", "focus": "Leadership", "meeting_frequency": "Bi-weekly"}),
            
            # Collaboration relationships
            Relationship("emp_001", "emp_002", "COLLABORATES_WITH", 
                        {"frequency": "Daily", "effectiveness": 4.5, "projects_shared": 2}),
            Relationship("emp_001", "emp_004", "COLLABORATES_WITH", 
                        {"frequency": "Daily", "effectiveness": 4.0, "projects_shared": 1}),
            Relationship("emp_002", "emp_005", "COLLABORATES_WITH", 
                        {"frequency": "Weekly", "effectiveness": 4.2, "projects_shared": 3}),
            Relationship("emp_001", "emp_003", "COLLABORATES_WITH", 
                        {"frequency": "Weekly", "effectiveness": 4.1, "projects_shared": 1}),
            
            # Career aspirations
            Relationship("emp_001", "pos_002", "ASPIRES_TO", 
                        {"timeline": "12 months", "readiness": 0.8, "development_plan": "Leadership training"}),
            Relationship("emp_002", "pos_003", "ASPIRES_TO", 
                        {"timeline": "18 months", "readiness": 0.7, "development_plan": "Management certification"}),
            Relationship("emp_004", "pos_001", "ASPIRES_TO", 
                        {"timeline": "24 months", "readiness": 0.4, "development_plan": "Senior developer skills"})
        ]
    
    def get_employee_network(self, employee_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get network around an employee"""
        if employee_id not in self.employees:
            return {"error": "Employee not found"}
        
        center_employee = self.employees[employee_id]
        connections = []
        
        # Find direct connections
        for rel in self.relationships:
            if rel.from_id == employee_id:
                connected_entity = self._get_entity_by_id(rel.to_id)
                if connected_entity:
                    connections.append({
                        "entity": connected_entity,
                        "relationship": {
                            "type": rel.relationship_type,
                            "properties": rel.properties,
                            "created_at": rel.created_at.isoformat()
                        },
                        "distance": 1
                    })
            elif rel.to_id == employee_id and rel.relationship_type not in ["HAS_SKILL", "WORKS_ON", "WORKED_ON"]:
                connected_entity = self._get_entity_by_id(rel.from_id)
                if connected_entity:
                    connections.append({
                        "entity": connected_entity,
                        "relationship": {
                            "type": rel.relationship_type,
                            "properties": rel.properties,
                            "created_at": rel.created_at.isoformat()
                        },
                        "distance": 1
                    })
        
        return {
            "center_employee": {
                "id": center_employee.id,
                "name": center_employee.name,
                "title": center_employee.title,
                "department": center_employee.department,
                "performance_rating": center_employee.performance_rating
            },
            "connections": connections,
            "total_connections": len(connections),
            "network_size": len(connections) + 1
        }
    
    def analyze_talent_pipeline(self, position_id: str) -> Dict[str, Any]:
        """Analyze talent pipeline for a position"""
        if position_id not in self.positions:
            return {"error": "Position not found"}
        
        position = self.positions[position_id]
        required_skills = position.required_skills
        
        # Find employees who aspire to this position
        aspirants = []
        for rel in self.relationships:
            if rel.relationship_type == "ASPIRES_TO" and rel.to_id == position_id:
                employee = self.employees[rel.from_id]
                skill_match = self._calculate_skill_match(rel.from_id, required_skills)
                aspirants.append({
                    "employee": {
                        "id": employee.id,
                        "name": employee.name,
                        "title": employee.title,
                        "performance_rating": employee.performance_rating
                    },
                    "readiness_score": rel.properties.get("readiness", 0),
                    "skill_match_percentage": skill_match,
                    "timeline": rel.properties.get("timeline", "Unknown"),
                    "development_plan": rel.properties.get("development_plan", "None")
                })
        
        # Find employees with matching skills
        skill_matches = []
        for emp_id, employee in self.employees.items():
            if emp_id not in [asp["employee"]["id"] for asp in aspirants]:
                skill_match = self._calculate_skill_match(emp_id, required_skills)
                if skill_match > 0.6:  # 60% skill match threshold
                    skill_matches.append({
                        "employee": {
                            "id": employee.id,
                            "name": employee.name,
                            "title": employee.title,
                            "performance_rating": employee.performance_rating
                        },
                        "skill_match_percentage": skill_match,
                        "potential_readiness": skill_match * 0.8  # Estimate based on skills
                    })
        
        return {
            "position": {
                "id": position.id,
                "title": position.title,
                "department": position.department,
                "required_skills": position.required_skills,
                "min_experience": position.min_experience
            },
            "direct_aspirants": sorted(aspirants, key=lambda x: x["readiness_score"], reverse=True),
            "skill_matches": sorted(skill_matches, key=lambda x: x["skill_match_percentage"], reverse=True),
            "pipeline_strength": len(aspirants) + len(skill_matches),
            "succession_risk": "Low" if len(aspirants) >= 2 else "High"
        }
    
    def identify_flight_risk_employees(self) -> List[Dict[str, Any]]:
        """Identify employees with high flight risk"""
        flight_risk_employees = []
        
        for emp_id, employee in self.employees.items():
            risk_factors = []
            risk_score = employee.flight_risk_score
            
            # Analyze risk factors
            if employee.tenure_months < 12:
                risk_factors.append("New employee (tenure < 12 months)")
            
            if employee.performance_rating < 3.5:
                risk_factors.append("Below average performance")
            
            # Check project involvement
            project_count = len([r for r in self.relationships 
                               if r.from_id == emp_id and r.relationship_type in ["WORKS_ON", "WORKED_ON"]])
            if project_count == 0:
                risk_factors.append("No recent project involvement")
            
            # Check mentorship
            has_mentor = any(r.to_id == emp_id and r.relationship_type == "MENTORS" for r in self.relationships)
            if not has_mentor and employee.tenure_months < 18:
                risk_factors.append("No mentorship support")
            
            # Check career progression
            has_aspiration = any(r.from_id == emp_id and r.relationship_type == "ASPIRES_TO" for r in self.relationships)
            if not has_aspiration:
                risk_factors.append("No clear career aspirations")
            
            if risk_score > 0.5:  # High risk threshold
                flight_risk_employees.append({
                    "employee": {
                        "id": employee.id,
                        "name": employee.name,
                        "title": employee.title,
                        "department": employee.department,
                        "tenure_months": employee.tenure_months,
                        "performance_rating": employee.performance_rating
                    },
                    "risk_score": risk_score,
                    "risk_factors": risk_factors,
                    "recommended_actions": self._get_retention_actions(risk_factors)
                })
        
        return sorted(flight_risk_employees, key=lambda x: x["risk_score"], reverse=True)
    
    def analyze_skill_gaps(self) -> Dict[str, Any]:
        """Analyze skill gaps across the organization"""
        skill_demand = {}
        skill_supply = {}
        
        # Calculate skill demand from positions
        for pos_id, position in self.positions.items():
            for skill in position.required_skills:
                skill_demand[skill] = skill_demand.get(skill, 0) + 1
        
        # Calculate skill supply from employees
        for rel in self.relationships:
            if rel.relationship_type == "HAS_SKILL":
                skill_obj = self.skills.get(rel.to_id)
                if skill_obj:
                    skill_name = skill_obj.name
                    proficiency = rel.properties.get("proficiency", 0)
                    if proficiency >= 3.0:  # Competent level
                        skill_supply[skill_name] = skill_supply.get(skill_name, 0) + 1
        
        # Calculate gaps
        skill_gaps = []
        for skill, demand in skill_demand.items():
            supply = skill_supply.get(skill, 0)
            if supply < demand:
                gap = demand - supply
                skill_gaps.append({
                    "skill": skill,
                    "demand": demand,
                    "supply": supply,
                    "gap": gap,
                    "gap_percentage": (gap / demand) * 100 if demand > 0 else 0
                })
        
        return {
            "critical_gaps": sorted(skill_gaps, key=lambda x: x["gap"], reverse=True),
            "total_skills_analyzed": len(skill_demand),
            "skills_with_gaps": len(skill_gaps),
            "overall_coverage": (len(skill_demand) - len(skill_gaps)) / len(skill_demand) * 100 if skill_demand else 100
        }
    
    def get_collaboration_insights(self) -> Dict[str, Any]:
        """Get collaboration insights across the organization"""
        collaboration_pairs = []
        department_collaboration = {}
        
        for rel in self.relationships:
            if rel.relationship_type == "COLLABORATES_WITH":
                emp1 = self.employees[rel.from_id]
                emp2 = self.employees[rel.to_id]
                
                collaboration_pairs.append({
                    "employee1": emp1.name,
                    "employee2": emp2.name,
                    "frequency": rel.properties.get("frequency", "Unknown"),
                    "effectiveness": rel.properties.get("effectiveness", 0),
                    "cross_department": emp1.department != emp2.department,
                    "projects_shared": rel.properties.get("projects_shared", 0)
                })
                
                # Track department collaboration
                dept_pair = tuple(sorted([emp1.department, emp2.department]))
                if dept_pair not in department_collaboration:
                    department_collaboration[dept_pair] = 0
                department_collaboration[dept_pair] += 1
        
        # Calculate collaboration metrics
        if collaboration_pairs:
            cross_dept_collaborations = len([c for c in collaboration_pairs if c["cross_department"]])
            avg_effectiveness = sum(c["effectiveness"] for c in collaboration_pairs) / len(collaboration_pairs)
        else:
            cross_dept_collaborations = 0
            avg_effectiveness = 0
        
        return {
            "total_collaborations": len(collaboration_pairs),
            "cross_department_collaborations": cross_dept_collaborations,
            "cross_department_percentage": (cross_dept_collaborations / len(collaboration_pairs)) * 100 if collaboration_pairs else 0,
            "average_effectiveness": round(avg_effectiveness, 2),
            "department_collaboration_matrix": department_collaboration,
            "top_collaborators": sorted(collaboration_pairs, key=lambda x: x["effectiveness"], reverse=True)[:3]
        }
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get organization-wide performance analytics"""
        performance_ratings = [emp.performance_rating for emp in self.employees.values()]
        
        return {
            "total_employees": len(self.employees),
            "average_performance": round(sum(performance_ratings) / len(performance_ratings), 2),
            "high_performers": len([r for r in performance_ratings if r >= 4.0]),
            "low_performers": len([r for r in performance_ratings if r < 3.0]),
            "performance_distribution": {
                "outstanding": len([r for r in performance_ratings if r >= 4.5]),
                "exceeds": len([r for r in performance_ratings if 4.0 <= r < 4.5]),
                "meets": len([r for r in performance_ratings if 3.0 <= r < 4.0]),
                "below": len([r for r in performance_ratings if 2.0 <= r < 3.0]),
                "unsatisfactory": len([r for r in performance_ratings if r < 2.0])
            }
        }
    
    def _get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID from any collection"""
        if entity_id in self.employees:
            emp = self.employees[entity_id]
            return {
                "type": "employee",
                "id": emp.id,
                "name": emp.name,
                "title": emp.title,
                "department": emp.department
            }
        elif entity_id in self.positions:
            pos = self.positions[entity_id]
            return {
                "type": "position",
                "id": pos.id,
                "title": pos.title,
                "department": pos.department
            }
        elif entity_id in self.skills:
            skill = self.skills[entity_id]
            return {
                "type": "skill",
                "id": skill.id,
                "name": skill.name,
                "category": skill.category
            }
        elif entity_id in self.projects:
            proj = self.projects[entity_id]
            return {
                "type": "project",
                "id": proj.id,
                "name": proj.name,
                "status": proj.status
            }
        return None
    
    def _calculate_skill_match(self, employee_id: str, required_skills: List[str]) -> float:
        """Calculate skill match percentage for an employee"""
        employee_skills = []
        for rel in self.relationships:
            if rel.from_id == employee_id and rel.relationship_type == "HAS_SKILL":
                skill_obj = self.skills.get(rel.to_id)
                if skill_obj and rel.properties.get("proficiency", 0) >= 3.0:
                    employee_skills.append(skill_obj.name)
        
        if not required_skills:
            return 0.0
        
        matches = len(set(employee_skills) & set(required_skills))
        return matches / len(required_skills)
    
    def _get_retention_actions(self, risk_factors: List[str]) -> List[str]:
        """Get recommended retention actions based on risk factors"""
        actions = []
        
        if "New employee (tenure < 12 months)" in risk_factors:
            actions.append("Assign mentor and create structured onboarding plan")
        
        if "Below average performance" in risk_factors:
            actions.append("Provide performance improvement plan and additional training")
        
        if "No recent project involvement" in risk_factors:
            actions.append("Assign to high-visibility project to increase engagement")
        
        if "No mentorship support" in risk_factors:
            actions.append("Pair with senior colleague for mentorship")
        
        if "No clear career aspirations" in risk_factors:
            actions.append("Conduct career development discussion and create growth plan")
        
        return actions

# Test function
def test_hr_model():
    """Test HR domain model"""
    print("=== Testing HR Model ===")
    
    hr_model = HRGraphModel()
    
    # Test 1: Employee Network Analysis
    print("\n1. Employee Network Analysis:")
    network = hr_model.get_employee_network("emp_001")
    if "error" not in network:
        print(f"Network for {network['center_employee']['name']}:")
        print(f"  Total connections: {network['total_connections']}")
        print(f"  Network size: {network['network_size']}")
        print(f"  Connection types: {[c['relationship']['type'] for c in network['connections']]}")
    else:
        print(f"Error: {network['error']}")
    
    # Test 2: Talent Pipeline Analysis
    print("\n2. Talent Pipeline Analysis:")
    pipeline = hr_model.analyze_talent_pipeline("pos_002")
    if "error" not in pipeline:
        print(f"Pipeline for {pipeline['position']['title']}:")
        print(f"  Direct aspirants: {len(pipeline['direct_aspirants'])}")
        print(f"  Skill matches: {len(pipeline['skill_matches'])}")
        print(f"  Pipeline strength: {pipeline['pipeline_strength']}")
        print(f"  Succession risk: {pipeline['succession_risk']}")
        
        if pipeline['direct_aspirants']:
            top_aspirant = pipeline['direct_aspirants'][0]
            print(f"  Top aspirant: {top_aspirant['employee']['name']} (readiness: {top_aspirant['readiness_score']})")
    else:
        print(f"Error: {pipeline['error']}")
    
    # Test 3: Flight Risk Analysis
    print("\n3. Flight Risk Analysis:")
    flight_risks = hr_model.identify_flight_risk_employees()
    print(f"High flight risk employees: {len(flight_risks)}")
    
    for risk_emp in flight_risks:
        print(f"  {risk_emp['employee']['name']}: {risk_emp['risk_score']} risk score")
        print(f"    Risk factors: {len(risk_emp['risk_factors'])}")
        print(f"    Recommended actions: {len(risk_emp['recommended_actions'])}")
    
    # Test 4: Skill Gap Analysis
    print("\n4. Skill Gap Analysis:")
    skill_gaps = hr_model.analyze_skill_gaps()
    print(f"Total skills analyzed: {skill_gaps['total_skills_analyzed']}")
    print(f"Skills with gaps: {skill_gaps['skills_with_gaps']}")
    print(f"Overall coverage: {skill_gaps['overall_coverage']:.1f}%")
    
    if skill_gaps['critical_gaps']:
        print("  Critical gaps:")
        for gap in skill_gaps['critical_gaps'][:3]:
            print(f"    {gap['skill']}: {gap['gap']} shortage ({gap['gap_percentage']:.1f}%)")
    
    # Test 5: Collaboration Insights
    print("\n5. Collaboration Insights:")
    collab_insights = hr_model.get_collaboration_insights()
    print(f"Total collaborations: {collab_insights['total_collaborations']}")
    print(f"Cross-department collaborations: {collab_insights['cross_department_percentage']:.1f}%")
    print(f"Average effectiveness: {collab_insights['average_effectiveness']}")
    
    if collab_insights['top_collaborators']:
        print("  Top collaborations:")
        for collab in collab_insights['top_collaborators']:
            print(f"    {collab['employee1']} ↔ {collab['employee2']}: {collab['effectiveness']} effectiveness")
    
    # Test 6: Performance Analytics
    print("\n6. Performance Analytics:")
    performance = hr_model.get_performance_analytics()
    print(f"Total employees: {performance['total_employees']}")
    print(f"Average performance: {performance['average_performance']}")
    print(f"High performers (≥4.0): {performance['high_performers']}")
    print(f"Low performers (<3.0): {performance['low_performers']}")
    print(f"Performance distribution: {performance['performance_distribution']}")

if __name__ == "__main__":
    test_hr_model()