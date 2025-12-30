#!/usr/bin/env python3
"""
Generate synthetic B2B enterprise artifacts with tenant isolation.

Artifacts:
- Documents
- Communications
- Processes
- Structured Records
- Decisions

Each artifact includes governance metadata:
- tenant_id
- access_control
- retention_rules
- deletion_capability
"""

import json
import random
import uuid
from pathlib import Path
import argparse

# -----------------------------
# CONFIG
# -----------------------------

CATEGORIES = {
    "documents": ["SOP", "Policy", "Manual", "Specification"],
    "communications": ["Email", "Support Ticket", "Chat Log"],
    "processes": ["Checklist", "Runbook", "Workflow"],
    "structured_records": ["CRM Entry", "Invoice", "System Log"],
    "decisions": ["Approval", "Escalation", "Resolution"],
}

PROCESS_OBJECTIVES = [
     "Ensure consistent execution of this workflow while reducing operational risk.",
    "Standardize how this process is performed to improve accountability and traceability.",
    "Enable teams to execute this process efficiently while ensuring compliance with internal standards.",
    "Reduce errors and delays by defining clear steps and escalation criteria for this process.",
    "Support reliable decision-making by ensuring this process is followed consistently."
]

ACCESS_ROLES = ["employee", "manager", "admin", "legal", "finance"]

RETENTION_POLICIES = [
    {"ttl_days": 90, "legal_hold": False},
    {"ttl_days": 365, "legal_hold": False},
    {"ttl_days": 1095, "legal_hold": True},
]

PROCESS_STEPS = {
    "Checklist": [
        "Condirm prerequisites are met",
        "Verify required inputs and documentaion",
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
        "Valiidate task completion and dependencies",
        "Close the workflow or escalete as needed exceptions"
    ],
}


# -----------------------------
# HELPERS
# -----------------------------

def random_governance():
    return {
        "access_control": {
            "roles": random.sample(ACCESS_ROLES, k=random.randint(1, 3))
        },
        "retention_rules": random.choice(RETENTION_POLICIES),
        "deletion_capability": {
            "deletable": True,
            "method": random.choice(["soft_delete", "hard_delete"]),
            "requires_approval": random.choice([True, False])
        }
    }

def generate_document_content(artifact_type: str, tenant_name: str) -> str:
    return f"""
{artifact_type} — {tenant_name}

Purpose
-------
This document defines internal standards and guidance used by {tenant_name}
to ensure consistent operations and regulatory compliance.

Scope
-----
Applies to all relevant teams, systems, and processes operating under {tenant_name}.

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

def generate_content(category: str, artifact_type: str, tenant_name: str) -> str:
    if category == "documents":
        return generate_document_content(artifact_type, tenant_name)

    if category == "processes":
        return (
            f"{artifact_type} — {tenant_name}\n\n"
            f"Objective\n"
            f"---------\n"
            f"Define a repeatable process used by {tenant_name} to ensure consistent execution, accountability, and timely escalation of issues.\n\n"
            f"Steps\n"
            f"-----\n"
            f"1. Identify required inputs and stakeholders\n"
            f"2. Execute the defined operational steps\n"
            f"3. Validate outputs and log results\n"
            f"4. Escalate issues if criteria are not met\n\n"
            f"Ownership\n"
            f"---------\n"
            f"Owned by the responsible operational team and reviewed periodically."
        )

    return (
        f"{artifact_type} for {tenant_name}. "
        f"This {category[:-1]} defines internal guidance, context, and operational details "
        f"used by the organization."
    )




def generate_artifact(tenant_id: str, tenant_name: str, category: str) -> dict:
    artifact_type = random.choice(CATEGORIES[category])
    return {
        "id": str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "category": category,
        "artifact_type": artifact_type,
        "content": generate_content(category, artifact_type, tenant_name),
        "governance": random_governance(),
    }


def derive_training_item(artifact: dict) -> dict:
    return {
        "tenant_id": artifact["tenant_id"],
        "category": artifact["category"],
        "instruction": "Summarize the artifact and propose next steps.",
        "context": artifact["content"],
        "response": (
            f"Summary: This {artifact['artifact_type']} outlines key operational information.\n"
            f"Next steps: Review for accuracy and ensure compliance with internal standards."
        ),
        "governance": artifact["governance"],
    }

# -----------------------------
# MAIN GENERATOR
# -----------------------------

def generate_b2b_dataset(
    tenants: int,
    artifacts_per_category: int,
    output_dir: Path,
    derive_training: bool,
    training_ratio: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_file = output_dir / "b2b_artifacts.jsonl"
    training_file = output_dir / "b2b_training_items.jsonl"

    tenants_info = [
        (f"tenant-{i:03d}", f"TenantCorp{i:03d}") for i in range(1, tenants + 1)
    ]

    with open(artifacts_file, "w") as af, open(training_file, "w") if derive_training else open(artifacts_file, "a") as tf:
        for tenant_id, tenant_name in tenants_info:
            for category in CATEGORIES:
                for _ in range(artifacts_per_category):
                    artifact = generate_artifact(tenant_id, tenant_name, category)
                    af.write(json.dumps(artifact) + "\n")

                    if derive_training and random.random() < training_ratio:
                        training_item = derive_training_item(artifact)
                        tf.write(json.dumps(training_item) + "\n")

    print("✅ B2B dataset generation complete")
    print(f"Artifacts saved to: {artifacts_file}")
    if derive_training:
        print(f"Training items saved to: {training_file}")

# -----------------------------
# PURGE UTILITY
# -----------------------------

def purge_tenant_data(output_dir: Path, tenant_id: str):
    for file_name in ["b2b_artifacts.jsonl", "b2b_training_items.jsonl"]:
        file_path = output_dir / file_name
        if not file_path.exists():
            continue

        remaining = []
        with open(file_path) as f:
            for line in f:
                if json.loads(line).get("tenant_id") != tenant_id:
                    remaining.append(line)

        with open(file_path, "w") as f:
            f.writelines(remaining)

        print(f"🧹 Purged {tenant_id} from {file_name}")

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate B2B synthetic enterprise data")
    parser.add_argument("--tenants", type=int, default=3)
    parser.add_argument("--artifacts-per-category", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--derive-training", action="store_true")
    parser.add_argument("--training-ratio", type=float, default=0.2)
    parser.add_argument("--purge-tenant", type=str)

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    if args.purge_tenant:
        purge_tenant_data(out_dir, args.purge_tenant)
    else:
        generate_b2b_dataset(
            tenants=args.tenants,
            artifacts_per_category=args.artifacts_per_category,
            output_dir=out_dir,
            derive_training=args.derive_training,
            training_ratio=args.training_ratio,
        )
