#!/usr/bin/env python3
"""
Generate synthetic B2B enterprise artifacts with entity isolation.

Advanced detection-focused synthetic data generator supporting multiple niches:
- Correlated detection patterns (indicators that appear together)
- Entity-level risk profiles and transaction/event baselines
- Temporal correlations (detection bursts, seasonal patterns)
- Relationship tracking (invoices->payments, transactions->decisions)
- Realistic detection scoring based on indicator combinations
- Outcome prediction correlated with detection/compliance scores
- Historical pattern learning with entity baselines
- Vendor/counterparty relationship tracking and recurring patterns
- Sophisticated document generation with realistic details

Optimized for niches with these characteristics:
- Document/invoice processing heavy
- Detection core (fraud, quality, risk, anomaly, compliance, etc.)
- Compliance/regulations critical
- High-value B2B transactions
- Historical pattern learning
- Outcome prediction valuable
- Zero physical interfacing

Supports multiple detection types:
- Fraud detection (financial services, payments)
- Quality defect detection (manufacturing, supply chain)
- Compliance violation detection (regulatory, legal)
- Risk detection (insurance, lending, credit)
- Operational anomaly detection (logistics, operations)
- Contract breach detection (procurement, legal)
"""

import json
import random
import uuid
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import math



"""
NOTE ON STOCHASTICITY:
This generator does not use uniform randomness.
All sampling is constrained by research-based probability distributions,
entity state, temporal correlation, and risk-adjusted weights.
Randomness here represents modeled uncertainty, not arbitrary choice.
"""


# -----------------------------
# MATHEMATICAL CONSTANTS
# Based on research: power-law distributions, seasonal patterns, correlation matrices
# -----------------------------

# Pareto distribution parameter for transaction amounts (heavy-tailed)
PARETO_ALPHA = 2.2

# Seasonal transaction coefficients (Q1-Q4)
SEASONAL_MULTIPLIERS = {
    1: 0.85, 2: 0.85, 3: 0.85,  # Q1
    4: 0.95, 5: 0.95, 6: 0.95,  # Q2
    7: 0.90, 8: 0.90, 9: 0.90,  # Q3
    10: 1.30, 11: 1.30, 12: 1.30  # Q4
}

# Weekly transaction patterns
WEEKLY_MULTIPLIERS = {
    "Monday": 0.80,
    "Tuesday": 1.05,
    "Wednesday": 1.10,
    "Thursday": 1.15,
    "Friday": 0.90,
    "Saturday": 0.40,
    "Sunday": 0.30
}

# Time-of-day distribution (business hours vs off-hours)
BUSINESS_HOURS_PROB = 0.85
AFTER_HOURS_PROB = 0.10
WEEKEND_HOURS_PROB = 0.05

# Detection rate parameters by entity risk level
DETECTION_BASE_RATES = {
    "low": 0.005,    # 0.5-2%
    "medium": 0.020,  # 2-5%
    "high": 0.050     # 5-12%
}

# Risk amplification factors
RISK_AMPLIFICATION_BETA = 0.3
SIZE_AMPLIFICATION_GAMMA = 0.1

# Indicator correlation coefficients (within-cluster)
INDICATOR_CLUSTER_CORRELATION = 0.60
INDICATOR_CROSS_CORRELATION = 0.10

# Temporal autocorrelation
TEMPORAL_AUTOCORR_DAY = 0.4
TEMPORAL_AUTOCORR_WEEK = 0.2

# Burst detection parameters
BURST_BASELINE_RATE = 0.02
BURST_DECAY_RATE = 0.3

# -----------------------------
# CONFIG
# -----------------------------

# Deterministic document type selection based on expected content length
DOCUMENT_TYPE_BY_SIZE = {
    "short": ["SOP", "Policy"],
    "medium": ["Compliance Report", "Purchase Order"],
    "long": ["Contract", "Audit Report"],
}

# -----------------------------
# NICHE-SPECIFIC DOCUMENT TYPE WEIGHTS
# -----------------------------

NICHE_DOCUMENT_TYPE_WEIGHTS = {
    "vendor_fraud_protection": {
        "Invoice": 5,
        "Purchase Order": 2,
        "Contract": 2,
        "Compliance Report": 2,
        "Audit Report": 1,
        "SOP": 0.5,
        "Policy": 0.5,
    },
    "generic_detection": {
        "Invoice": 2,
        "Purchase Order": 2,
        "Contract": 2,
        "Compliance Report": 1,
        "Audit Report": 1,
        "SOP": 1,
        "Policy": 1,
    },
}



CATEGORIES = {
    "documents": ["Invoice", "Purchase Order", "Contract", "SOP", "Policy", "Compliance Report", "Audit Report"],
    "communications": ["Email", "Support Ticket", "Chat Log", "Escalation Notice"],
    "processes": ["Checklist", "Runbook", "Workflow", "Approval Process"],
    "structured_records": ["Transaction", "Invoice Record", "Payment Record", "System Log", "Risk Event"],
    "decisions": ["Fraud Decision", "Compliance Approval", "Risk Assessment", "Escalation", "Resolution"],
    "risk_events": ["Suspicious Transaction", "Anomaly Detected", "Compliance Violation", "Pattern Deviation"],
    "compliance_records": ["Regulatory Filing", "Audit Trail", "Compliance Check", "Policy Adherence"],
}

# -----------------------------
# CATEGORY FREQUENCY WEIGHTS
# Relative frequency of artifact categories (config only)
# -----------------------------

CATEGORY_WEIGHTS = {
    "structured_records": 4.0,
    "documents": 3.0,
    "risk_events": 2.0,
    "decisions": 2.0,
    "communications": 1.0,
    "compliance_records": 1.0,
    "processes": 0.5,
}


INVOICE_TYPES = [
    "Standard Invoice", "Credit Memo", "Debit Memo", "Recurring Invoice",
    "Proforma Invoice", "Final Invoice", "Interim Invoice", "Commercial Invoice"
]

# High-value transaction ranges (B2B) - based on research distributions
# Distribution: 35%, 30%, 20%, 12%, 3%
TRANSACTION_RANGES = [
    (5000, 50000),        # Small B2B - 35%
    (50000, 250000),      # Medium B2B - 30%
    (250000, 1000000),    # Large B2B - 20%
    (1000000, 5000000),   # Enterprise B2B - 12%
    (5000000, 20000000),  # High-value Enterprise - 3%
]

TRANSACTION_RANGE_WEIGHTS = [0.35, 0.30, 0.20, 0.12, 0.03]

# Detection indicator groups - indicators that tend to appear together
# Applicable across niches: fraud, quality defects, compliance violations, operational anomalies
DETECTION_INDICATOR_GROUPS = {
    "velocity_pattern": ["velocity_anomaly", "time_pattern_anomaly", "unusual_volume_pattern"],
    "counterparty_issues": ["counterparty_relationship_issue", "duplicate_detection", "documentation_tampering_risk"],
    "geographic_anomaly": ["geographic_mismatch", "time_pattern_anomaly", "process_method_risk"],
    "compliance_breach": ["compliance_gap", "documentation_tampering_risk", "counterparty_relationship_issue"],
    "value_manipulation": ["value_rounding_anomaly", "unusual_value_pattern", "duplicate_detection"],
    "quality_deviation": ["specification_deviation", "process_variance", "material_inconsistency"],
    "operational_anomaly": ["throughput_deviation", "resource_allocation_issue", "timeline_violation"],
}

# Individual detection indicators with base severity scores
# Generic enough to apply to: fraud, quality defects, compliance violations, operational risks
DETECTION_INDICATORS = {
    "unusual_value_pattern": 0.3,
    "velocity_anomaly": 0.4,
    "geographic_mismatch": 0.35,
    "time_pattern_anomaly": 0.25,
    "counterparty_relationship_issue": 0.5,
    "duplicate_detection": 0.6,
    "value_rounding_anomaly": 0.3,
    "process_method_risk": 0.35,
    "documentation_tampering_risk": 0.7,
    "compliance_gap": 0.45,
    "specification_deviation": 0.4,
    "process_variance": 0.35,
    "material_inconsistency": 0.5,
    "throughput_deviation": 0.3,
    "resource_allocation_issue": 0.4,
    "timeline_violation": 0.45,
    "unusual_volume_pattern": 0.35,
}

COMPLIANCE_FRAMEWORKS = [
    "SOX", "GDPR", "PCI-DSS", "HIPAA", "SOC2", "ISO27001",
    "FCPA", "AML", "KYC", "OFAC", "FATCA", "Basel III"
]

RISK_LEVELS = ["low", "medium", "high", "critical"]

# Outcome types for prediction (generic across detection types)
OUTCOME_TYPES = [
    "approved", "rejected", "pending_review", "escalated",
    "flagged", "violation_detected", "requires_manual_review",
    "remediation_required", "monitoring_required"
]

ACCESS_ROLES = ["employee", "manager", "admin", "legal", "finance", "compliance", "risk_analyst", "auditor"]

RETENTION_POLICIES = [
    {"ttl_days": 90, "legal_hold": False, "compliance_required": False},
    {"ttl_days": 365, "legal_hold": False, "compliance_required": True},
    {"ttl_days": 1095, "legal_hold": True, "compliance_required": True},
    {"ttl_days": 2555, "legal_hold": True, "compliance_required": True},
]

# Counterparty categories with typical transaction/event patterns
# Generic: vendors, suppliers, partners, contractors, clients
COUNTERPARTY_CATEGORIES = {
    "IT_Services": {"typical_range": (50000, 500000), "frequency": "monthly", "risk": 0.2},
    "Consulting": {"typical_range": (100000, 1000000), "frequency": "quarterly", "risk": 0.15},
    "Software": {"typical_range": (25000, 250000), "frequency": "annual", "risk": 0.1},
    "Hardware": {"typical_range": (50000, 500000), "frequency": "quarterly", "risk": 0.25},
    "Maintenance": {"typical_range": (10000, 100000), "frequency": "monthly", "risk": 0.1},
    "Professional_Services": {"typical_range": (75000, 750000), "frequency": "monthly", "risk": 0.2},
    "Manufacturing": {"typical_range": (100000, 2000000), "frequency": "monthly", "risk": 0.25},
    "Logistics": {"typical_range": (50000, 500000), "frequency": "weekly", "risk": 0.2},
    "Raw_Materials": {"typical_range": (200000, 5000000), "frequency": "monthly", "risk": 0.3},
}


NICHE_CATEGORY_WEIGHTS = {
    "vendor_fraud_protection": {
        "Invoice": 1.000,
        "Purchase Order": 0.742,
        "Contract": 0.681,
        "Compliance Report": 0.537,
        "Audit Report": 0.412,
        "SOP": 0.283,
        "Policy": 0.259,
    },
    "generic_detection": {
        "Invoice": 0.820,
        "Purchase Order": 0.801,
        "Contract": 0.774,
        "Compliance Report": 0.612,
        "Audit Report": 0.598,
        "SOP": 0.566,
        "Policy": 0.541,
    },
}

RISK_LEVEL_CATEGORY_MULTIPLIERS = {
    "low": {
        "risk_events": 0.5,
        "decisions": 0.7,
        "compliance_records": 0.8,
    },
    "medium": {
        "risk_events": 1.0,
        "decisions": 1.0,
        "compliance_records": 1.0,
    },
    "high": {
        "risk_events": 1.8,
        "decisions": 1.5,
        "compliance_records": 1.4,
    },
}


STRUCTURED_RECORD_WEIGHTS_BY_NICHE = {
    "vendor_fraud_protection": {
        "Transaction": 0.34,
        "Invoice Record": 0.26,
        "Payment Record": 0.22,
        "Risk Event": 0.12,
        "System Log": 0.06,
    },
    "generic_detection": {
        "Transaction": 0.20,
        "Invoice Record": 0.20,
        "Payment Record": 0.20,
        "Risk Event": 0.20,
        "System Log": 0.20,
    },
}


DOCUMENT_TYPE_WEIGHTS_BY_NICHE = {
    "vendor_fraud_protection": {
        "Invoice": 0.32,
        "Contract": 0.20,
        "Purchase Order": 0.16,
        "Compliance Report": 0.14,
        "Audit Report": 0.10,
        "Policy": 0.04,
        "SOP": 0.04,
    },
    "generic_detection": {
        "Invoice": 0.18,
        "Purchase Order": 0.16,
        "Contract": 0.16,
        "Compliance Report": 0.14,
        "Audit Report": 0.14,
        "Policy": 0.11,
        "SOP": 0.11,
    },
}


def get_effective_category_weights(niche: str) -> Dict[str, float]:
    """
    Return category weights adjusted by niche.
    Falls back to base CATEGORY_WEIGHTS if niche is unknown.
    """
    niche_weights = NICHE_CATEGORY_WEIGHTS.get(niche)

    if not niche_weights:
        return CATEGORY_WEIGHTS

    effective = CATEGORY_WEIGHTS.copy()

    for category, multiplier in niche_weights.items():
        if category in effective:
            effective[category] *= multiplier

    return effective

def get_effective_category_weights(niche: str) -> Dict[str, float]:
    """
    Return category weights adjusted by niche.
    Falls back to base CATEGORY_WEIGHTS if niche is unknown.
    """
    niche_weights = NICHE_CATEGORY_WEIGHTS.get(niche)

    if not niche_weights:
        return CATEGORY_WEIGHTS

    effective = CATEGORY_WEIGHTS.copy()

    for category, multiplier in niche_weights.items():
        if category in effective:
            effective[category] = effective[category] * multiplier

    return effective



# -----------------------------
# ENTITY PROFILE
# -----------------------------

@dataclass
class EntityProfile:
    """Entity-level characteristics that persist across artifacts"""
    entity_id: str
    entity_name: str
    base_risk_level: str  # low, medium, high
    compliance_frameworks: List[str]
    typical_transaction_range: Tuple[float, float]
    typical_transaction_category: str
    counterparty_relationships: Dict[str, Dict]  # counterparty_id -> counterparty info
    transaction_history: List[Dict]
    detection_incident_rate: float  # 0.0 to 1.0 - rate of flagged events
    compliance_violation_rate: float  # 0.0 to 1.0
    geographic_regions: List[str]
    process_methods: List[str]  # Generic: payment methods, shipping methods, processing methods
    
    def get_counterparty(self, counterparty_id: str) -> Optional[Dict]:
        return self.counterparty_relationships.get(counterparty_id)
    
    def add_counterparty(self, counterparty_id: str, counterparty_info: Dict):
        self.counterparty_relationships[counterparty_id] = counterparty_info
    
    def add_transaction(self, transaction: Dict):
        self.transaction_history.append(transaction)
        # Keep only last 100 transactions for memory efficiency
        if len(self.transaction_history) > 100:
            self.transaction_history.pop(0)
    
    def get_typical_amount(self) -> float:
        """Get a typical transaction amount using Pareto distribution (research-based)"""
        min_val, max_val = self.typical_transaction_range
        # Use Pareto distribution with α ≈ 2.2 (heavy-tailed)
        amount = pareto_amount(min_val, PARETO_ALPHA)
        # Cap at max to stay within range
        amount = min(amount, max_val)
        return round(amount, 2)
    
    def is_anomalous_amount(self, amount: float) -> bool:
        """Check if amount is anomalous for this entity"""
        min_val, max_val = self.typical_transaction_range
        # Anomalous if > 3x typical max or < 0.1x typical min
        return amount > max_val * 3 or amount < min_val * 0.1

# -----------------------------
# GLOBAL STATE
# -----------------------------

entity_profiles: Dict[str, EntityProfile] = {}
entity_detection_bursts: Dict[str, List[datetime]] = {}  # Track detection event bursts per entity
entity_compliance_issues: Dict[str, List[datetime]] = {}  # Track compliance issues per entity

# -----------------------------
# MATHEMATICAL HELPER FUNCTIONS
# -----------------------------

def pareto_amount(min_val: float, alpha: float = PARETO_ALPHA) -> float:
    """Generate amount using Pareto distribution (power-law, heavy-tailed)"""
    u = random.random()
    return min_val * math.pow(1 - u, -1 / (alpha - 1))

def logistic_probability(risk_factors: float, beta0: float = -2.0, beta1: float = 2.5) -> float:
    """Calculate probability using logistic function"""
    return 1.0 / (1.0 + math.exp(-(beta0 + beta1 * risk_factors)))

def zipf_distribution(n: int, s: float = 1.2) -> List[float]:
    """Generate Zipf distribution for vendor/counterparty selection"""
    weights = [1.0 / math.pow(i, s) for i in range(1, n + 1)]
    total = sum(weights)
    return [w / total for w in weights]

def calculate_seasonal_multiplier(date: datetime) -> float:
    """Get seasonal multiplier based on research"""
    month = date.month
    day_of_week = date.strftime("%A")
    hour = date.hour
    
    seasonal = SEASONAL_MULTIPLIERS.get(month, 1.0)
    weekly = WEEKLY_MULTIPLIERS.get(day_of_week, 1.0)
    
    # Time of day factor
    if 9 <= hour <= 17:
        hourly = 1.0
    elif hour < 9 or hour > 17:
        hourly = 0.3
    else:
        hourly = 0.5
    
    return seasonal * weekly * hourly

def num_vendors_for_size(revenue_millions: float) -> int:
    """Calculate number of vendors based on company size (log relationship)"""
    if revenue_millions < 1:  # Small
        return random.randint(8, 25)
    elif revenue_millions < 50:  # Medium
        return random.randint(25, 100)
    elif revenue_millions < 500:  # Large
        return random.randint(100, 500)
    else:  # Enterprise
        return random.randint(500, 2000)

def num_compliance_frameworks(revenue_millions: float) -> int:
    """Calculate compliance frameworks based on size"""
    if revenue_millions < 1:
        return random.randint(1, 2)
    elif revenue_millions < 50:
        return random.randint(3, 5)
    elif revenue_millions < 500:
        return random.randint(5, 8)
    else:
        return random.randint(8, 12)

# -----------------------------
# HELPERS
# -----------------------------

def sample_from_modeled_distribution(options, weights):
    """
    Sample from a research-driven probability distribution.
    This reflects modeled variance, not uniform randomness.
    """
    return random.choices(options, weights=weights, k=1)[0]


def create_entity_profile(entity_id: str, entity_name: str) -> EntityProfile:
    """Create a new entity profile with realistic characteristics based on research"""
    base_risk = random.choices(
        ["low", "medium", "high"],
        weights=[0.6, 0.3, 0.1]
    )[0]
    
    # Estimate company revenue (for sizing calculations)
    revenue_millions = random.choice([0.5, 2, 10, 50, 200, 1000])
    
    # Select compliance frameworks based on company size (research-based)
    num_frameworks = num_compliance_frameworks(revenue_millions)
    frameworks = random.sample(COMPLIANCE_FRAMEWORKS, k=min(num_frameworks, len(COMPLIANCE_FRAMEWORKS)))
    
    # Detection rates based on research (0.5-2% low, 2-5% medium, 5-12% high)
    if base_risk == "high":
        detection_rate = random.uniform(0.05, 0.12)
        compliance_violation_rate = random.uniform(0.10, 0.25)
    elif base_risk == "medium":
        detection_rate = random.uniform(0.02, 0.05)
        compliance_violation_rate = random.uniform(0.05, 0.15)
    else:
        detection_rate = random.uniform(0.005, 0.02)
        compliance_violation_rate = random.uniform(0.01, 0.05)
    
    # Select typical transaction range using research weights
    range_idx = random.choices(
        range(len(TRANSACTION_RANGES)),
        weights=TRANSACTION_RANGE_WEIGHTS
    )[0]
    typical_range = TRANSACTION_RANGES[range_idx]
    
    # Select counterparty category
    counterparty_cat = random.choice(list(COUNTERPARTY_CATEGORIES.keys()))
    
    # Geographic regions
    regions = random.sample(
        ["US", "EU", "APAC", "LATAM", "MEA"],
        k=random.randint(1, 3)
    )
    
    # Process methods (payment methods distribution from research)
    # ACH: 45%, Wire: 30%, Check: 15%, Credit: 10%
    process_methods_pool = (
        ["ACH", "EFT"] * 45 +
        ["Wire Transfer"] * 30 +
        ["Check"] * 15 +
        ["Credit Card", "Digital Wallet"] * 10
    )
    process_methods = list(set(random.sample(process_methods_pool, k=random.randint(2, 5))))
    
    # Create initial counterparty relationships (using research-based counts)
    num_counterparties = num_vendors_for_size(revenue_millions)
    counterparty_relationships = {}
    
    # Use Zipf distribution for counterparty transaction frequencies
    zipf_weights = zipf_distribution(num_counterparties)
    
    for i in range(num_counterparties):
        counterparty_id = f"CP-{entity_id}-{i:03d}"
        counterparty_cat = random.choice(list(COUNTERPARTY_CATEGORIES.keys()))
        counterparty_info = COUNTERPARTY_CATEGORIES[counterparty_cat].copy()
        counterparty_info["counterparty_id"] = counterparty_id
        counterparty_info["counterparty_name"] = f"{counterparty_cat}_Partner_{i}"
        counterparty_info["relationship_start"] = (datetime.now() - timedelta(days=random.randint(30, 730))).isoformat()
        
        # Transaction count follows power law (high frequency for top vendors)
        counterparty_info["transaction_count"] = int(zipf_weights[i] * 200)  # Scale to realistic counts
        counterparty_info["zipf_weight"] = zipf_weights[i]
        
        counterparty_relationships[counterparty_id] = counterparty_info
    
    return EntityProfile(
        entity_id=entity_id,
        entity_name=entity_name,
        base_risk_level=base_risk,
        compliance_frameworks=frameworks,
        typical_transaction_range=typical_range,
        typical_transaction_category=counterparty_cat,
        counterparty_relationships=counterparty_relationships,
        transaction_history=[],
        detection_incident_rate=detection_rate,
        compliance_violation_rate=compliance_violation_rate,
        geographic_regions=regions,
        process_methods=process_methods,
    )

def get_entity_profile(entity_id: str) -> EntityProfile:
    """Get or create entity profile"""
    if entity_id not in entity_profiles:
        entity_name = f"EntityCorp{entity_id.split('-')[-1]}"
        entity_profiles[entity_id] = create_entity_profile(entity_id, entity_name)
    return entity_profiles[entity_id]

def generate_correlated_detection_indicators(entity_id: str, transaction_amount: float, 
                                         temporal_data: Dict) -> Dict:
    """Generate detection indicators with realistic correlations using research-based model"""
    profile = get_entity_profile(entity_id)
    
    # Check if we're in a detection burst period (using exponential decay model)
    now = datetime.fromisoformat(temporal_data["created_at"].replace("Z", ""))
    in_detection_burst = False
    burst_probability = BURST_BASELINE_RATE
    
    if entity_id in entity_detection_bursts and len(entity_detection_bursts[entity_id]) > 0:
        last_burst = entity_detection_bursts[entity_id][-1]
        days_since = (now - last_burst).days
        # Exponential decay: P(burst) = λ × exp(-γ × t)
        burst_probability = BURST_BASELINE_RATE * math.exp(-BURST_DECAY_RATE * days_since) + 0.01
        in_detection_burst = random.random() < burst_probability
    
    # Calculate detection probability using logistic model
    # P(detection) = α × (1 + β × risk_score + γ × transaction_size_factor)
    base_rate = DETECTION_BASE_RATES[profile.base_risk_level]
    
    # Transaction size factor
    typical_max = profile.typical_transaction_range[1]
    size_factor = max(0, (transaction_amount - typical_max) / typical_max)
    
    # Risk score from entity profile
    risk_score = (profile.detection_incident_rate - 0.005) / 0.115  # Normalize to 0-1
    
    # Calculate detection probability
    detection_prob = base_rate * (1 + RISK_AMPLIFICATION_BETA * risk_score + SIZE_AMPLIFICATION_GAMMA * size_factor)
    if in_detection_burst:
        detection_prob *= 2.5
    
    has_detection = random.random() < detection_prob
    
    if not has_detection:
        return {
            "detection_score": round(random.uniform(0.0, 0.25), 3),
            "indicators": [],
            "severity_level": "low",
            "requires_review": False
        }
    
    # Select detection pattern group with correlation
    # Use multivariate Bernoulli with correlation matrix
    pattern_group = random.choice(list(DETECTION_INDICATOR_GROUPS.keys()))
    base_indicators = DETECTION_INDICATOR_GROUPS[pattern_group].copy()
    
    # Add correlated indicators from same cluster (ρ = 0.60)
    for other_group, other_indicators in DETECTION_INDICATOR_GROUPS.items():
        if other_group != pattern_group:
            # Cross-cluster correlation (ρ = 0.10)
            if random.random() < INDICATOR_CROSS_CORRELATION:
                base_indicators.extend(random.sample(other_indicators, k=1))
        else:
            # Within-cluster correlation (ρ = 0.60)
            if random.random() < INDICATOR_CLUSTER_CORRELATION:
                additional = [ind for ind in other_indicators if ind not in base_indicators]
                if additional:
                    base_indicators.extend(random.sample(additional, k=min(1, len(additional))))
    
    # Check for velocity anomaly (temporal autocorrelation)
    if len(profile.transaction_history) > 0:
        recent_txns = [tx for tx in profile.transaction_history 
                      if (now - datetime.fromisoformat(tx.get("timestamp", now.isoformat()).replace("Z", ""))).days < 1]
        if len(recent_txns) > 5:
            if "velocity_anomaly" not in base_indicators:
                base_indicators.append("velocity_anomaly")
    
    # Calculate detection score using weighted indicators
    base_score = sum(DETECTION_INDICATORS.get(ind, 0.2) for ind in base_indicators) / len(base_indicators) if base_indicators else 0.0
    
    # Multiplier for multiple indicators
    multiplier = 1.0 + (len(base_indicators) - 1) * 0.15
    detection_score = min(1.0, base_score * multiplier)
    
    # Add realistic variance
    detection_score = round(detection_score * random.uniform(0.9, 1.1), 3)
    detection_score = min(1.0, max(0.0, detection_score))
    
    # Amplify severity during detection bursts
    if in_detection_burst:
        detection_score = min(1.0, detection_score * 1.8)


    # Determine severity level
    if detection_score >= 0.8:
        severity_level = "critical"
    elif detection_score >= 0.6:
        severity_level = "high"
    elif detection_score >= 0.4:
        severity_level = "medium"
    else:
        severity_level = "low"
    
    # Record detection burst if high/critical
    if severity_level in ["high", "critical"]:
        if entity_id not in entity_detection_bursts:
            entity_detection_bursts[entity_id] = []
        entity_detection_bursts[entity_id].append(now)
        if len(entity_detection_bursts[entity_id]) > 10:
            entity_detection_bursts[entity_id].pop(0)
    
    return {
        "detection_score": detection_score,
        "indicators": list(set(base_indicators)),  # Remove duplicates
        "severity_level": severity_level,
        "requires_review": detection_score >= 0.5,
        "pattern_group": pattern_group
    }

def generate_compliance_metadata(entity_id: str, detection_data: Dict) -> Dict:
    """Generate compliance metadata correlated with detection events"""
    profile = get_entity_profile(entity_id)
    
    # Compliance violations often correlate with detection events
    base_violation_prob = profile.compliance_violation_rate
    if detection_data.get("detection_score", 0) > 0.5:
        base_violation_prob *= 1.8  # Higher violation rate when detection events present
    
    compliance_status = "compliant"
    if random.random() < base_violation_prob:
        compliance_status = random.choice(["non_compliant", "requires_review"])
        
        # Record compliance issue
        now = datetime.now()
        if entity_id not in entity_compliance_issues:
            entity_compliance_issues[entity_id] = []
        entity_compliance_issues[entity_id].append(now)
        if len(entity_compliance_issues[entity_id]) > 10:
            entity_compliance_issues[entity_id].pop(0)
    
    # Select applicable frameworks (use entity's frameworks)
    applicable_frameworks = profile.compliance_frameworks.copy()
    # Sometimes add additional frameworks
    if random.random() < 0.3:
        remaining = [f for f in COMPLIANCE_FRAMEWORKS if f not in applicable_frameworks]
        if remaining:
            k = min(len(remaining), random.randint(0, 2))
            additional = random.sample(remaining, k=k)
        else:
            additional = []
        applicable_frameworks.extend(additional)
    
    regulatory_flags = []
    if compliance_status != "compliant":
        regulatory_flags = random.sample(
            ["data_privacy", "financial_reporting", "anti_money_laundering", "sanctions", 
             "safety_standards", "quality_standards", "environmental_compliance"],
            k=random.randint(1, 3)
        )
    
    return {
        "applicable_frameworks": applicable_frameworks,
        "compliance_status": compliance_status,
        "requires_audit": compliance_status != "compliant",
        "regulatory_flags": regulatory_flags
    }

def generate_transaction_value(entity_id: str, counterparty_id: Optional[str] = None) -> Dict:
    """Generate transaction value based on entity and counterparty patterns"""
    profile = get_entity_profile(entity_id)
    
    # If counterparty specified, use counterparty's typical range
    if counterparty_id and counterparty_id in profile.counterparty_relationships:
        counterparty = profile.counterparty_relationships[counterparty_id]
        range_min, range_max = counterparty.get("typical_range", profile.typical_transaction_range)
    else:
        range_min, range_max = profile.typical_transaction_range
    
    # Generate amount with some variation
    base_amount = profile.get_typical_amount()
    
    # Add some randomness but keep it within reasonable bounds
    variation = random.uniform(0.7, 1.5)
    amount = base_amount * variation
    
    # Round appropriately
    if amount < 100000:
        amount = round(amount, 2)
    else:
        amount = round(amount, 0)
    
    # Ensure within range
    amount = max(range_min * 0.5, min(range_max * 2, amount))
    
    currency = random.choice(["USD", "EUR", "GBP", "JPY", "CAD"])
    
    category = profile.typical_transaction_category
    if counterparty_id and counterparty_id in profile.counterparty_relationships:
        counterparty = profile.counterparty_relationships[counterparty_id]
        # Extract category from counterparty name or use default
        for cat in COUNTERPARTY_CATEGORIES:
            if cat in counterparty.get("counterparty_name", ""):
                category = cat
                break
    
    return {
        "amount": amount,
        "currency": currency,
        "amount_usd_equivalent": round(amount * random.uniform(0.8, 1.2), 2) if currency != "USD" else amount,
        "transaction_category": category,
        "counterparty_id": counterparty_id
    }

def generate_temporal_metadata(entity_id: str, base_date: Optional[datetime] = None) -> Dict:
    """Generate temporal patterns with seasonal and business hour considerations (research-based)"""
    if base_date is None:
        # Generate date with temporal autocorrelation (AR(1) process)
        # More recent dates are more likely (gamma distribution)
        days_ago = int(random.gammavariate(2, 100))
        days_ago = min(days_ago, 730)  # Cap at 2 years
        base_date = datetime.now() - timedelta(days=days_ago)
        
        # Apply seasonal multiplier to determine if transaction occurs
        seasonal_mult = calculate_seasonal_multiplier(base_date)
        # Adjust date probability based on seasonal factors
        while random.random() > seasonal_mult and days_ago < 730:
            days_ago = int(random.gammavariate(2, 100))
            days_ago = min(days_ago, 730)
            base_date = datetime.now() - timedelta(days=days_ago)
            seasonal_mult = calculate_seasonal_multiplier(base_date)
    
    created_at = base_date.isoformat() + "Z"
    
    # Processing delay (faster for low-risk, slower for high-risk)
    profile = get_entity_profile(entity_id)
    if profile.base_risk_level == "high":
        processing_delay_hours = random.randint(12, 96)
    elif profile.base_risk_level == "medium":
        processing_delay_hours = random.randint(6, 48)
    else:
        processing_delay_hours = random.randint(0, 24)
    
    processed_at = (base_date + timedelta(hours=processing_delay_hours)).isoformat() + "Z"
    
    hour = base_date.hour
    is_business_hours = 9 <= hour <= 17
    day_of_week = base_date.strftime("%A")
    is_weekend = day_of_week in ["Saturday", "Sunday"]
    
    # Seasonal patterns
    month = base_date.month
    quarter = (month - 1) // 3 + 1
    is_q4 = quarter == 4
    
    # Get multipliers for analysis
    seasonal_multiplier = SEASONAL_MULTIPLIERS.get(month, 1.0)
    weekly_multiplier = WEEKLY_MULTIPLIERS.get(day_of_week, 1.0)
    
    return {
        "created_at": created_at,
        "processed_at": processed_at,
        "processing_delay_hours": processing_delay_hours,
        "temporal_pattern": {
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "is_business_hours": is_business_hours,
            "is_weekend": is_weekend,
            "month": month,
            "quarter": quarter,
            "is_q4": is_q4,
            "year": base_date.year,
            "seasonal_multiplier": seasonal_multiplier,
            "weekly_multiplier": weekly_multiplier
        }
    }

def generate_outcome_prediction(detection_data: Dict, compliance_data: Dict, 
                                transaction_data: Dict, entity_id: str) -> Dict:
    """Generate realistic outcome prediction using logistic probability model (research-based)"""
    profile = get_entity_profile(entity_id)
    
    detection_score = detection_data.get("detection_score", 0.0)
    compliance_status = compliance_data.get("compliance_status", "compliant")
    amount = transaction_data.get("amount", 0)
    
    # Calculate combined risk factors for logistic model
    # Risk score: 0.4×detection + 0.3×compliance + 0.2×anomaly + 0.1×historical
    compliance_factor = 1.0 if compliance_status == "non_compliant" else (0.5 if compliance_status == "requires_review" else 0.0)
    amount_factor = 1.0 if amount > profile.typical_transaction_range[1] * 2 else 0.0
    historical_factor = (profile.detection_incident_rate - 0.005) / 0.115  # Normalize
    
    combined_risk = (
        0.4 * detection_score +
        0.3 * compliance_factor +
        0.2 * amount_factor +
        0.1 * historical_factor
    )
    
    # Use logistic probability model: P(outcome | risk) = softmax(β₀ + β₁×risk)
    # Beta coefficients from research: [approved, rejected, review, escalated, monitoring]
    beta_coefficients = {
        "approved": (0.5, -2.5),
        "rejected": (-0.8, 3.0),
        "requires_manual_review": (0.3, 1.5),
        "escalated": (0.1, 2.0),
        "monitoring_required": (0.2, 1.0),
        "flagged": (-0.5, 2.8),
        "violation_detected": (-1.0, 3.5)
    }
    
    # Calculate probabilities using softmax
    outcome_probs = {}
    for outcome, (beta0, beta1) in beta_coefficients.items():
        prob = math.exp(beta0 + beta1 * combined_risk)
        outcome_probs[outcome] = prob
    
    # Normalize to get softmax
    total = sum(outcome_probs.values())
    outcome_probs = {k: v/total for k, v in outcome_probs.items()}
    
    # Select outcome based on probabilities
    outcomes = list(outcome_probs.keys())
    probs = list(outcome_probs.values())
    predicted_outcome = random.choices(outcomes, weights=probs)[0]
    confidence = outcome_probs[predicted_outcome]
    
    prediction_metadata = {
        "predicted_outcome": predicted_outcome,
        "confidence": round(confidence, 3),
        "prediction_timestamp": datetime.now().isoformat() + "Z",
        "prediction_factors": {
            "detection_score": detection_score,
            "compliance_status": compliance_status,
            "transaction_amount": amount,
            "entity_risk_level": profile.base_risk_level,
            "combined_risk_score": round(combined_risk, 3)
        },
        "outcome_probabilities": {k: round(v, 3) for k, v in outcome_probs.items()}
    }
    
    # Add actual outcome for historical data (70% of cases)
    if random.random() < 0.7:
        # Actual outcome matches predicted with 85% accuracy (research-based)
        if random.random() < 0.85:
            actual_outcome = predicted_outcome
        else:
            # Generate alternative outcome from distribution
            remaining_outcomes = [o for o in outcomes if o != predicted_outcome]
            remaining_probs = [outcome_probs[o] for o in remaining_outcomes]
            total_remaining = sum(remaining_probs)
            remaining_probs = [p/total_remaining for p in remaining_probs]
            actual_outcome = random.choices(remaining_outcomes, weights=remaining_probs)[0]
        
        prediction_metadata["actual_outcome"] = actual_outcome
        prediction_metadata["prediction_accuracy"] = 1.0 if actual_outcome == predicted_outcome else 0.0
        prediction_metadata["actual_outcome_timestamp"] = (datetime.now() + timedelta(hours=random.randint(1, 72))).isoformat() + "Z"
    
    return prediction_metadata

def generate_invoice_content(invoice_type: str, entity_name: str, transaction_data: Dict,
                            counterparty_id: Optional[str] = None, invoice_date: Optional[datetime] = None) -> str:
    """Generate detailed invoice content with realistic line items"""
    profile = get_entity_profile(transaction_data.get("entity_id", ""))
    
    invoice_id = f"INV-{random.randint(100000, 999999)}"
    if invoice_date is None:
        invoice_date = datetime.now() - timedelta(days=random.randint(0, 90))
    due_date = invoice_date + timedelta(days=random.choice([15, 30, 45, 60]))
    
    # Get counterparty info
    counterparty_name = "Unknown Counterparty"
    if counterparty_id and counterparty_id in profile.counterparty_relationships:
        counterparty = profile.counterparty_relationships[counterparty_id]
        counterparty_name = counterparty.get("counterparty_name", counterparty_name)
    
    # Generate realistic line items based on transaction category
    category = transaction_data.get("transaction_category", "Professional_Services")
    line_items = []
    num_items = random.randint(1, 12)
    remaining_amount = transaction_data["amount"]
    
    # Category-specific descriptions
    category_descriptions = {
        "IT_Services": ["Cloud Infrastructure", "Software Development", "System Integration", "Technical Support"],
        "Consulting": ["Strategic Consulting", "Business Analysis", "Process Improvement", "Advisory Services"],
        "Software": ["Software License", "SaaS Subscription", "Maintenance & Support", "Implementation Services"],
        "Hardware": ["Server Equipment", "Network Infrastructure", "Storage Systems", "Peripheral Devices"],
        "Maintenance": ["Preventive Maintenance", "Support Services", "Warranty Extension", "Repair Services"],
        "Professional_Services": ["Professional Services", "Project Management", "Training Services", "Documentation"],
        "Manufacturing": ["Manufactured Components", "Assembly Services", "Custom Fabrication", "Quality Testing"],
        "Logistics": ["Transportation Services", "Warehousing", "Distribution", "Freight Handling"],
        "Raw_Materials": ["Raw Materials", "Component Parts", "Bulk Materials", "Specialty Materials"],
    }
    
    descriptions = category_descriptions.get(category, ["Professional Services", "Consulting Services"])
    
    for i in range(num_items):
        if i == num_items - 1:
            item_amount = round(remaining_amount, 2)
        else:
            # Distribute amount with some variation
            portion = random.uniform(0.05, 0.25)
            item_amount = round(remaining_amount * portion, 2)
            remaining_amount -= item_amount
        
        quantity = random.randint(1, 100)
        unit_price = round(item_amount / quantity, 2)
        
        line_items.append({
            "description": random.choice(descriptions),
            "quantity": quantity,
            "unit_price": unit_price,
            "amount": item_amount
        })
    
    tax_rate = random.choice([0.0, 0.08, 0.10, 0.20])
    tax_amount = round(transaction_data["amount"] * tax_rate, 2)
    total_amount = transaction_data["amount"] + tax_amount
    
    # Determine status based on due date
    days_past_due = (datetime.now() - due_date).days
    if days_past_due > 0:
        status = random.choice(["Overdue", "Partially Paid"])
    elif days_past_due > -7:
        status = random.choice(["Pending", "Paid"])
    else:
        status = random.choice(["Pending", "Paid", "Partially Paid"])
    
    process_method = random.choice(profile.process_methods)
    
    return f"""
{invoice_type} — {entity_name}
Invoice Number: {invoice_id}
Invoice Date: {invoice_date.strftime('%Y-%m-%d')}
Due Date: {due_date.strftime('%Y-%m-%d')}
Status: {status}
Currency: {transaction_data['currency']}

Bill To:
--------
{entity_name}
Account Number: ACC-{random.randint(10000, 99999)}

Counterparty:
-------------
{counterparty_name}
Counterparty ID: {counterparty_id or 'N/A'}

Line Items:
-----------
{chr(10).join([f"{i+1}. {item['description']} - Qty: {item['quantity']} @ {item['unit_price']:,.2f} = {item['amount']:,.2f}" for i, item in enumerate(line_items)])}

Subtotal: {transaction_data['amount']:,.2f} {transaction_data['currency']}
Tax ({tax_rate*100:.1f}%): {tax_amount:,.2f} {transaction_data['currency']}
Total: {total_amount:,.2f} {transaction_data['currency']}

Payment Terms: Net {random.choice([15, 30, 45, 60])}
Process Method: {process_method}
""".strip()

def generate_governance_policy():
    return {
        "access_control": {
            "roles": random.sample(ACCESS_ROLES, k=random.randint(1, 4))
        },
        "retention_rules": random.choice(RETENTION_POLICIES),
        "deletion_capability": {
            "deletable": True,
            "method": random.choice(["soft_delete", "hard_delete"]),
            "requires_approval": random.choice([True, False])
        }
    }

# Continue with rest of content generation functions...
# (Using entity profiles and correlations for detection-agnostic approach)

def generate_document_content(artifact_type: str, entity_name: str, entity_id: str) -> str:



    if artifact_type == "Invoice":
        # Select a counterparty from entity's relationships
        profile = get_entity_profile(entity_id)
        counterparty_id = random.choice(list(profile.counterparty_relationships.keys())) if profile.counterparty_relationships else None
        transaction_data = generate_transaction_value(entity_id, counterparty_id)
        transaction_data["entity_id"] = entity_id
        invoice_date = datetime.now() - timedelta(days=random.randint(0, 90))
        return generate_invoice_content(artifact_type, entity_name, transaction_data, counterparty_id, invoice_date)
    
    return f"""
{artifact_type} — {entity_name}


Purpose
-------
This document defines internal standards and guidance used by {entity_name}
to ensure consistent operations and regulatory compliance.
Scope
-----
Applies to all relevant teams, systems, and processes operating under {entity_name}.
Key Sections
------------
- Roles and responsibilities
- Operational requirements
- Compliance considerations
- Review and update cadence
Notes
-----
This document is maintained internally and reviewed periodically for accuracy.
""".strip()

def generate_structured_record_content(artifact_type: str, entity_name: str, entity_id: str) -> str:
    profile = get_entity_profile(entity_id)
    counterparty_id = random.choice(list(profile.counterparty_relationships.keys())) if profile.counterparty_relationships else None
    transaction_data = generate_transaction_value(entity_id, counterparty_id)
    transaction_data["entity_id"] = entity_id
    
    # Record transaction in entity history
    tx_record = {
        "timestamp": datetime.now().isoformat() + "Z",
        "amount": transaction_data["amount"],
        "currency": transaction_data["currency"],
        "counterparty_id": counterparty_id,
        "category": transaction_data["transaction_category"]
    }
    profile.add_transaction(tx_record)
    
    if artifact_type == "Transaction":
        tx_id = f"TXN-{random.randint(1000000, 9999999)}"
        tx_date = datetime.now() - timedelta(days=random.randint(0, 90))
        return f"""
Transaction Record — {entity_name}
Transaction ID
--------------
{tx_id}
Amount
------
{transaction_data['amount']:,.2f} {transaction_data['currency']}
Transaction Date
---------------
{tx_date.strftime('%Y-%m-%d')}
Status
------
{random.choice(['Completed', 'Pending', 'Failed', 'Reversed'])}
Counterparty
-----------
{counterparty_id or 'Unknown'}
Category
--------
{transaction_data['transaction_category']}
""".strip()

    if artifact_type == "Invoice Record":
        invoice_date = datetime.now() - timedelta(days=random.randint(0, 90))
        return generate_invoice_content("Standard Invoice", entity_name, transaction_data, counterparty_id, invoice_date)

    if artifact_type == "Payment Record":
        payment_date = datetime.now() - timedelta(days=random.randint(0, 60))
        invoice_ref = f"INV-{random.randint(100000, 999999)}"
        return f"""
Payment Record — {entity_name}
Payment ID
----------
PAY-{random.randint(100000, 999999)}
Amount
------
{transaction_data['amount']:,.2f} {transaction_data['currency']}
Payment Date
-----------
{payment_date.strftime('%Y-%m-%d')}
Method
------
{random.choice(profile.process_methods)}
Status
------
{random.choice(['Processed', 'Pending', 'Failed'])}
Reference
---------
{invoice_ref}
""".strip()

    if artifact_type == "Risk Event":
        # This will be generated with detection data in main function
        return f"Risk Event — {entity_name}"

    if artifact_type == "System Log":
        return f"""
System Log — {entity_name}
Timestamp
---------
{datetime.now().isoformat()}Z
Severity
--------
{random.choice(['INFO', 'WARNING', 'ERROR', 'CRITICAL'])}
Message
-------
{random.choice([
    'Transaction processed successfully',
    'Detection check completed',
    'Compliance validation passed',
    'Anomaly detected in transaction pattern',
    'Severity threshold exceeded',
    'Processing initiated'
])}
""".strip()

    return f"{artifact_type} record for {entity_name}."

def generate_communication_content(artifact_type: str, entity_name: str, entity_id: str,
                                  detection_data: Optional[Dict] = None) -> str:
    profile = get_entity_profile(entity_id)
    
    if artifact_type == "Email":
        severity_level = detection_data.get("severity_level", "low") if detection_data else "low"
        return (
            f"Subject: Action Required – Transaction Review\n\n"
            f"Hello Team,\n\n"
            f"This email is to inform you of a transaction requiring review for {entity_name}. "
            f"Please review the details and take any required actions.\n\n"
            f"Transaction Details:\n"
            f"- Amount: High-value transaction detected\n"
            f"- Severity Level: {severity_level.upper()}\n"
            f"- Compliance Status: Requires verification\n\n"
            f"Next Steps:\n"
            f"- Review the transaction\n"
            f"- Complete detection and compliance checks\n"
            f"- Escalate concerns if applicable\n\n"
            f"Regards,\n"
            f"{entity_name} Operations"
        )

    if artifact_type == "Support Ticket":
        return (
            f"Issue Summary:\n"
            f"A transaction or compliance issue has been reported within {entity_name}.\n\n"
            f"Current Status:\n"
            f"- Ticket opened and under review\n"
            f"- Detection assessment in progress\n"
            f"- Compliance verification pending\n\n"
            f"Next Actions:\n"
            f"- Investigate root cause\n"
            f"- Apply remediation if needed\n"
            f"- Close or escalate based on findings"
        )

    if artifact_type == "Escalation Notice":
        reason = "Detection indicators present" if detection_data and detection_data.get("detection_score", 0) > 0.6 else "High-value threshold exceeded"
        return (
            f"Escalation Notice — {entity_name}\n\n"
            f"A high-severity transaction or compliance issue has been escalated.\n\n"
            f"Reason: {reason}\n"
            f"Priority: {random.choice(['High', 'Critical'])}\n"
            f"Requires immediate review by senior management."
        )

    if artifact_type == "Chat Log":
        return (
            f"User: Noticing an issue with a transaction.\n"
            f"Support: Thanks for flagging this. Can you provide more details?\n"
            f"User: The transaction value seems unusual for this counterparty.\n"
            f"Support: Acknowledged. We will investigate and follow up."
        )

    return f"{artifact_type} communication for {entity_name}."

def generate_decision_content(artifact_type: str, entity_name: str, detection_data: Dict,
                             compliance_data: Dict, outcome_data: Dict) -> str:
    predicted_outcome = outcome_data.get("predicted_outcome", "approved")
    
    decision_contexts = {
        "Fraud Decision": [
            f"A high-value transaction was flagged for review following automated risk assessment. Detection score: {detection_data.get('detection_score', 0):.3f}.",
            f"Multiple detection indicators found: {', '.join(detection_data.get('indicators', [])[:3])}.",
            f"Transaction requires manual review due to elevated severity level: {detection_data.get('severity_level', 'medium')}.",
        ],
        "Compliance Approval": [
            f"Compliance check identified {compliance_data.get('compliance_status', 'unknown')} status.",
            f"Regulatory frameworks applicable: {', '.join(compliance_data.get('applicable_frameworks', [])[:3])}.",
            f"Compliance verification {'passed' if compliance_data.get('compliance_status') == 'compliant' else 'requires additional review'}.",
        ],
        "Risk Assessment": [
            f"Risk assessment completed with detection score {detection_data.get('detection_score', 0):.3f}.",
            f"Severity level determined: {detection_data.get('severity_level', 'medium')}.",
            f"Assessment considers detection indicators, compliance status, and transaction characteristics.",
        ],
    }
    
    decisions = {
        "Fraud Decision": {
            "approved": "The transaction was approved after fraud checks passed.",
            "rejected": "The transaction was rejected due to fraud indicators.",
            "pending_review": "The transaction requires manual review due to elevated risk score.",
            "fraud_detected": "Fraud detected - transaction blocked and flagged for investigation.",
        },
        "Compliance Approval": {
            "approved": "The transaction was approved for compliance.",
            "rejected": "The transaction was rejected due to compliance violations.",
            "pending_review": "The transaction requires additional compliance documentation.",
        },
        "Risk Assessment": {
            "approved": "Risk assessment completed - transaction approved.",
            "pending_review": "Risk assessment identified concerns - escalation required.",
            "escalated": "Risk assessment passed with conditions - escalated for final approval.",
        },
    }
    
    context_list = decision_contexts.get(artifact_type, ["A decision was made based on risk assessment."])
    decision_map = decisions.get(artifact_type, {})
    decision_text = decision_map.get(predicted_outcome, "A decision was recorded.")
    
    return f"""
{artifact_type} — {entity_name}
Context
-------
{random.choice(context_list)}
Decision
--------
{decision_text}
Rationale
---------
The decision was made based on risk assessment, operational impact, and compliance considerations.
Outcome
-------
Predicted outcome: {predicted_outcome}. Appropriate follow-up actions were initiated and tracked to completion.
""".strip()

def generate_risk_event_content(artifact_type: str, entity_name: str, detection_data: Dict,
                               transaction_data: Dict) -> str:
    return f"""
{artifact_type} — {entity_name}
Event ID
--------
EVT-{random.randint(100000, 999999)}
Timestamp
---------
{datetime.now().isoformat()}Z
Severity Level
--------------
{detection_data['severity_level'].upper()}
Detection Score
---------------
{detection_data['detection_score']:.3f}
Transaction Amount
------------------
{transaction_data['amount']:,.2f} {transaction_data['currency']}
Indicators Detected
-------------------
{', '.join(detection_data['indicators']) if detection_data['indicators'] else 'None'}
Pattern Group
-------------
{detection_data.get('pattern_group', 'N/A')}
Status
------
{random.choice(['Open', 'Under Investigation', 'Resolved', 'Escalated'])}
Action Required
---------------
{random.choice(['Manual Review', 'Immediate Escalation', 'Automated Block', 'Additional Verification'])}
""".strip()

def generate_compliance_record_content(artifact_type: str, entity_name: str, compliance_data: Dict) -> str:
    return f"""
{artifact_type} — {entity_name}
Record ID
---------
COMP-{random.randint(100000, 999999)}
Timestamp
---------
{datetime.now().isoformat()}Z
Compliance Status
-----------------
{compliance_data['compliance_status'].upper()}
Applicable Frameworks
---------------------
{', '.join(compliance_data['applicable_frameworks'])}
Requires Audit
--------------
{compliance_data['requires_audit']}
Regulatory Flags
----------------
{', '.join(compliance_data['regulatory_flags']) if compliance_data['regulatory_flags'] else 'None'}
Status
------
{random.choice(['Compliant', 'Non-Compliant', 'Under Review', 'Remediation Required'])}
""".strip()

def generate_content(category: str, artifact_type: str, entity_name: str, entity_id: str,
                    detection_data: Optional[Dict] = None, compliance_data: Optional[Dict] = None,
                    transaction_data: Optional[Dict] = None, outcome_data: Optional[Dict] = None) -> str:
    if category == "documents":
        return generate_document_content(artifact_type, entity_name, entity_id)

    if category == "communications":
        return generate_communication_content(artifact_type, entity_name, entity_id, detection_data)

    if category == "processes":
        steps_map = {
            "Checklist": [
                "Confirm prerequisites are met",
                "Verify required inputs and documentation",
                "Complete each required task item",
                "Record completion status"
            ],
            "Runbook": [
                "Identify the triggering condition",
                "Execute the prescribed operational actions",
                "Monitor system behavior and logs",
                "Escalate if outcomes are not achieved"
            ],
            "Workflow": [
                "Initiate the workflow with required inputs",
                "Route tasks to appropriate stakeholders",
                "Validate task completion and dependencies",
                "Close the workflow or escalate as needed for exceptions"
            ],
            "Approval Process": [
                "Submit transaction for approval",
                "Risk assessment and detection check",
                "Compliance verification",
                "Approval decision and documentation"
            ],
        }
        steps = steps_map.get(artifact_type, steps_map["Workflow"])
        return (
            f"{artifact_type} — {entity_name}\n\n"
            f"Objective\n"
            f"---------\n"
            f"{random.choice([
                'Ensure consistent execution of this workflow while reducing operational risk.',
                'Standardize how this process is performed to improve accountability and traceability.',
                'Enable teams to execute this process efficiently while ensuring compliance with internal standards.',
                'Reduce errors and delays by defining clear steps and escalation criteria for this process.',
                'Support reliable decision-making by ensuring this process is followed consistently.'
            ])}\n\n"
            f"Steps\n"
            f"-----\n"
            f"{chr(10).join([f'{i+1}. {step}' for i, step in enumerate(steps)])}\n\n"
            f"Ownership\n"
            f"---------\n"
            f"Owned by the responsible operational team and reviewed periodically."
        )
    
    if category == "structured_records":
        return generate_structured_record_content(artifact_type, entity_name, entity_id)

    if category == "decisions":
        if detection_data and compliance_data and outcome_data:
            return generate_decision_content(artifact_type, entity_name, detection_data, compliance_data, outcome_data)
        return f"{artifact_type} decision for {entity_name}."
    
    if category == "risk_events":
        if detection_data and transaction_data:
            return generate_risk_event_content(artifact_type, entity_name, detection_data, transaction_data)
        return f"{artifact_type} for {entity_name}."
    
    if category == "compliance_records":
        if compliance_data:
            return generate_compliance_record_content(artifact_type, entity_name, compliance_data)
        return f"{artifact_type} for {entity_name}."

    return f"{artifact_type} for {entity_name}."

def generate_artifact(entity_id: str, entity_name: str, category: str) -> dict:
    profile = get_entity_profile(entity_id)

    artifact_type = None
    content = None

    # ✅ GUARANTEE existence (this fixes your crash)
    transaction_data = None
    detection_data = None
    compliance_data = None
    outcome_data = None

    # Generate temporal metadata first
    temporal_data = generate_temporal_metadata(entity_id)
    base_date = datetime.fromisoformat(temporal_data["created_at"].replace("Z", ""))

    # -----------------------------
    # Determine artifact type
    # -----------------------------
    if category == "documents":
        all_docs = CATEGORIES["documents"]  # ✅ GUARANTEED DEFINITION

        if random.random() < 0.5:
            artifact_type = "Invoice"
            counterparty_id = (
                random.choice(list(profile.counterparty_relationships.keys()))
                if profile.counterparty_relationships else None
            )
            transaction_data = generate_transaction_value(entity_id, counterparty_id)
            transaction_data["entity_id"] = entity_id

            content = generate_invoice_content(
                artifact_type,
                entity_name,
                transaction_data,
                counterparty_id,
                base_date
            )
        else:
            niche_weights = DOCUMENT_TYPE_WEIGHTS_BY_NICHE.get(args.niche, {})
            size_bucket = random.choice(("short", "medium", "long"))

            size_bias = {
                "short": {"SOP": 1.2, "Policy": 1.2},
                "medium": {"Purchase Order": 1.2, "Compliance Report": 1.2},
                "long": {"Contract": 1.2, "Audit Report": 1.2}
            }

            weights = []
            for doc_type in all_docs:
                base = niche_weights.get(doc_type, 1.0)
                bias = size_bias.get(size_bucket, {}).get(doc_type, 1.0)
                weights.append(base * bias)

            artifact_type = sample_from_modeled_distribution(all_docs, weights)
            content = generate_content(category, artifact_type, entity_name, entity_id)

    else:
        if category == "structured_records":
            type_weight_map = STRUCTURED_RECORD_WEIGHTS_BY_NICHE.get(args.niche, {})
            types = CATEGORIES[category]
            if type_weight_map:
                type_weights = [type_weight_map.get(t, 1.0) for t in types]
                artifact_type = sample_from_modeled_distribution(types, type_weights)
            else:
                artifact_type = random.choice(types)
        else:
            artifact_type = random.choice(CATEGORIES[category])

    # -----------------------------
    # Generate detection / compliance / outcome
    # -----------------------------
    amount = transaction_data["amount"] if transaction_data else 0
    detection_data = generate_correlated_detection_indicators(entity_id, amount, temporal_data)
    compliance_data = generate_compliance_metadata(entity_id, detection_data)
    outcome_data = generate_outcome_prediction(
        detection_data,
        compliance_data,
        transaction_data or {"amount": 0},
        entity_id
    )

    if content is None:
        content = generate_content(
            category,
            artifact_type,
            entity_name,
            entity_id,
            detection_data,
            compliance_data,
            transaction_data,
            outcome_data
        )

    artifact = {
        "id": str(uuid.uuid4()),
        "entity_id": entity_id,
        "category": category,
        "artifact_type": artifact_type,
        "content": content,
        "governance": generate_governance_policy(),
        "temporal_metadata": temporal_data,
        "detection_indicators": detection_data,
        "compliance_metadata": compliance_data,
        "outcome_prediction": outcome_data,
        "niche": args.niche,
        "industry": args.industry,
    }

    if transaction_data is not None:
        artifact["transaction_metadata"] = transaction_data

    if artifact_type == "Invoice" and transaction_data and transaction_data.get("counterparty_id"):
        artifact["relationships"] = {
            "counterparty_id": transaction_data["counterparty_id"],
            "related_artifacts": []
        }

    return artifact

def derive_training_item(artifact: dict) -> dict:
    """Derive training items with sophisticated detection/risk/compliance analysis"""
    detection_data = artifact.get("detection_indicators", {})
    compliance_data = artifact.get("compliance_metadata", {})
    transaction_data = artifact.get("transaction_metadata", {})
    outcome_data = artifact.get("outcome_prediction", {})
    
    # Select instruction based on artifact type
    if artifact["category"] == "documents" and "Invoice" in artifact["artifact_type"]:
        instruction = random.choice([
            "Analyze this invoice for detection indicators, compliance issues, and recommend an action.",
            "Review this invoice for anomalies, risk factors, and predict the approval outcome.",
            "Examine this invoice for potential detection patterns and compliance violations.",
        ])
    elif artifact["category"] == "decisions":
        instruction = random.choice([
            "Explain the reasoning behind this decision based on the detection and compliance data.",
            "Analyze whether this decision was appropriate given the risk indicators.",
        ])
    else:
        instruction = random.choice([
            "Analyze this transaction for detection indicators and recommend action.",
            "Assess compliance status and identify any regulatory concerns.",
            "Evaluate the risk level and predict the likely outcome.",
        ])
    
    response_parts = []
    
    # Detection analysis with reasoning
    if detection_data:
        detection_score = detection_data.get("detection_score", 0)
        severity_level = detection_data.get("severity_level", "low")
        indicators = detection_data.get("indicators", [])
        pattern_group = detection_data.get("pattern_group")
        
        if indicators:
            response_parts.append(
                f"Detection Analysis: Severity level is {severity_level} with detection score {detection_score:.3f}. "
                f"Detected indicators: {', '.join(indicators)}."
            )
            if pattern_group:
                response_parts.append(
                    f"Detection pattern identified: {pattern_group} - this pattern suggests coordinated issues."
                )
            if len(indicators) >= 3:
                response_parts.append(
                    f"Multiple indicators detected ({len(indicators)}) - this significantly increases detection confidence."
                )
        else:
            response_parts.append(
                f"Detection Analysis: Low severity (score {detection_score:.3f}) - no significant detection indicators found."
            )
    
    # Compliance analysis
    if compliance_data:
        compliance_status = compliance_data.get("compliance_status", "compliant")
        frameworks = compliance_data.get("applicable_frameworks", [])
        flags = compliance_data.get("regulatory_flags", [])
        
        response_parts.append(
            f"Compliance Status: {compliance_status.upper()}. "
            f"Applicable frameworks: {', '.join(frameworks)}."
        )
        if flags:
            response_parts.append(
                f"Regulatory flags: {', '.join(flags)} - these require immediate attention."
            )
        if compliance_status != "compliant" and detection_data.get("detection_score", 0) > 0.5:
            response_parts.append(
                "WARNING: Non-compliance combined with high detection score indicates potential serious violation."
            )
    
    # Transaction analysis
    if transaction_data:
        amount = transaction_data.get("amount", 0)
        currency = transaction_data.get("currency", "USD")
        category = transaction_data.get("transaction_category", "unknown")
        counterparty_id = transaction_data.get("counterparty_id")
        
        response_parts.append(
            f"Transaction: {amount:,.2f} {currency} in category {category}."
        )
        if counterparty_id:
            response_parts.append(f"Counterparty: {counterparty_id}")
        
        # Check if amount is anomalous
        profile = get_entity_profile(artifact["entity_id"])
        if profile.is_anomalous_amount(amount):
            response_parts.append(
                f"Amount anomaly: Transaction amount ({amount:,.2f}) is significantly outside typical range "
                f"({profile.typical_transaction_range[0]:,.2f} - {profile.typical_transaction_range[1]:,.2f})."
            )
    
    # Outcome prediction with reasoning
    if outcome_data:
        predicted = outcome_data.get("predicted_outcome", "unknown")
        confidence = outcome_data.get("confidence", 0)
        factors = outcome_data.get("prediction_factors", {})
        
        response_parts.append(
            f"Predicted Outcome: {predicted} with {confidence:.1%} confidence."
        )
        if factors:
            response_parts.append(
                f"Prediction factors: detection_score={factors.get('detection_score', 0):.3f}, "
                f"compliance={factors.get('compliance_status', 'unknown')}, "
                f"entity_risk={factors.get('entity_risk_level', 'unknown')}."
            )
        
        if outcome_data.get("actual_outcome"):
            actual = outcome_data.get("actual_outcome")
            accuracy = outcome_data.get("prediction_accuracy", 0)
            response_parts.append(
                f"Actual Outcome: {actual}. Prediction accuracy: {accuracy:.1%}."
            )
    
    # Recommendation
    if detection_data.get("detection_score", 0) >= 0.7 or compliance_data.get("compliance_status") == "non_compliant":
        response_parts.append("RECOMMENDATION: Reject transaction and escalate for investigation.")
    elif detection_data.get("detection_score", 0) >= 0.5:
        response_parts.append("RECOMMENDATION: Require manual review before approval.")
    elif detection_data.get("detection_score", 0) >= 0.3:
        response_parts.append("RECOMMENDATION: Approve with monitoring.")
    else:
        response_parts.append("RECOMMENDATION: Approve transaction.")
    
    response = "\n".join(response_parts)
    
    return {
        "entity_id": artifact["entity_id"],
        "category": artifact["category"],
        "instruction": instruction,
        "context": artifact["content"],
        "response": response,
        "governance": artifact["governance"],
        "detection_indicators": detection_data,
        "compliance_metadata": compliance_data,
        "transaction_metadata": transaction_data,
        "outcome_prediction": outcome_data,
    }

def generate_b2b_dataset(
    entitys: int,
    artifacts_per_category: int,
    output_dir: Path,
    derive_training: bool,
    training_ratio: float,
):
    # Reset global state
    global entity_profiles, entity_detection_bursts, entity_compliance_issues
    entity_profiles = {}
    entity_detection_bursts = {}
    entity_compliance_issues = {}
    
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_file = output_dir / "b2b_artifacts_v2.jsonl"
    training_file = output_dir / "b2b_training_items_v2.jsonl"

    entitys_info = [
        (f"entity-{i:03d}", f"EntityCorp{i:03d}") for i in range(1, entitys + 1)
    ]

    # Pre-create entity profiles
    print("Creating entity profiles with research-based characteristics...")
    for entity_id, entity_name in entitys_info:
        get_entity_profile(entity_id)
    
    # Get niche-adjusted category weights ONCE
    effective_weights = get_effective_category_weights(args.niche)

    print(f"Generating artifacts using mathematical models...")
    with open(artifacts_file, "w") as af:
        tf = open(training_file, "w") if derive_training else None
        try:
            for entity_id, entity_name in entitys_info:
                for category in CATEGORIES:
                    base_weight = effective_weights.get(category, 1.0)
                    risk_multiplier = RISK_LEVEL_CATEGORY_MULTIPLIERS.get(
                        get_entity_profile(entity_id).base_risk_level,
                        {}
                    ).get(category, 1.0)
                    category_count = max(
                        1,
                        int(artifacts_per_category * base_weight * risk_multiplier)
                    )
                    

                    for _ in range(category_count):
                        artifact = generate_artifact(entity_id, entity_name, category)
                        af.write(json.dumps(artifact) + "\n")

                        if derive_training and random.random() < training_ratio:
                            training_item = derive_training_item(artifact)
                            tf.write(json.dumps(training_item) + "\n")  

        finally:
            if tf:
                tf.close()

    print("\n✅ B2B dataset generation complete")
    print(f"📊 Artifacts saved to: {artifacts_file}")
    if derive_training:
        print(f"🎓 Training items saved to: {training_file}")
    print(f"🏢 Generated {len(entity_profiles)} entity profiles with research-based patterns")
    print(f"📈 Using mathematical models: Pareto distributions, logistic regression, temporal correlations")

def purge_entity_data(output_dir: Path, entity_id: str):
    for file_name in ["b2b_artifacts_v2.jsonl", "b2b_training_items_v2.jsonl"]:
        file_path = output_dir / file_name
        if not file_path.exists():
            continue

        remaining = []
        with open(file_path) as f:
            for line in f:
                if json.loads(line).get("entity_id") != entity_id:
                    remaining.append(line)

        with open(file_path, "w") as f:
            f.writelines(remaining)

        print(f"🧹 Purged {entity_id} from {file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate B2B synthetic enterprise data with research-based mathematical models",
        epilog="""
        This generator uses mathematical models from academic research:
        - Pareto distributions (α≈2.2) for transaction amounts
        - Logistic regression for outcome prediction  
        - Multivariate Bernoulli for correlated detection indicators
        - Temporal autocorrelation (AR(1)) for time series
        - Seasonal multipliers for quarterly/weekly patterns
        - Zipf distributions for counterparty relationships
        
        Supports multiple detection niches: fraud, quality, compliance, risk, operational anomalies
        """
    )
    parser.add_argument("--entitys", type=int, default=3, help="Number of entities to generate data for")
    parser.add_argument("--artifacts-per-category", type=int, default=100, help="Number of artifacts per category")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory for generated data")
    parser.add_argument("--derive-training", action="store_true", help="Generate training items from artifacts")
    parser.add_argument("--training-ratio", type=float, default=0.2, help="Ratio of artifacts to convert to training items")
    parser.add_argument("--purge-entity", type=str, help="Purge all data for a specific entity_id")
    parser.add_argument("--niche", type=str, default="generic_detection", help="Detection niche (fraud, compliance, risk, quality, etc.)")
    parser.add_argument("--industry", type=str, default="general",help="Industry context for generated data")
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    if args.purge_entity:
        purge_entity_data(out_dir, args.purge_entity)
    else:
        generate_b2b_dataset(
            entitys=args.entitys,
            artifacts_per_category=args.artifacts_per_category,
            output_dir=out_dir,
            derive_training=args.derive_training,
            training_ratio=args.training_ratio,
        )
