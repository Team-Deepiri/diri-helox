#!/usr/bin/env python3
"""
Generate synthetic B2B enterprise artifacts with entity isolation.

Artifacts:
- Documents
- Communications
- Processes
- Structured Records
- Decisions

Each artifact includes governance metadata:
- entity_id
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

def generate_document_content(artifact_type: str, entity_name: str) -> str:
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

def generate_structured_record_content(artifact_type: str, entity_name: str) -> str:
    if artifact_type == "CRM Entry":
        return f"""
CRM Entry — {entity_name}

Account Status
--------------
Active

Last Contact
------------
2025-01-12

Notes
-----
Customer engagement is stable. Follow up scheduled for next review cycle.
""".strip()

    if artifact_type == "Invoice":
        return f"""
Invoice — {entity_name}

Invoice ID
----------
INV-{random.randint(10000,99999)}

Amount
------
$4,250.00

Due Date
--------
2025-02-15

Status
------
Pending payment
""".strip()

    if artifact_type == "System Log":
        return f"""
System Log — {entity_name}

Timestamp
---------
2025-01-15T14:32:10Z

Severity
--------
WARNING

Message
-------
Transient failure detected during workflow execution.
""".strip()


def generate_communication_content(artifact_type: str, entity_name: str) -> str:
    if artifact_type == "Email":
        return (
            f"Subject: Action Required – Operational Update\n\n"
            f"Hello Team,\n\n"
            f"This email is to inform you of an operational update affecting {entity_name}. "
            f"Please review the details and take any required actions.\n\n"
            f"Next Steps:\n"
            f"- Review the update\n"
            f"- Confirm any required changes\n"
            f"- Escalate concerns if applicable\n\n"
            f"Regards,\n"
            f"{entity_name} Operations"
        )

    if artifact_type == "Support Ticket":
        return (
            f"Issue Summary:\n"
            f"A system or process issue has been reported within {entity_name}.\n\n"
            f"Current Status:\n"
            f"- Ticket opened and under review\n\n"
            f"Next Actions:\n"
            f"- Investigate root cause\n"
            f"- Apply remediation if needed\n"
            f"- Close or escalate based on findings"
        )

    if artifact_type == "Chat Log":
        return (
            f"User: Noticing an issue with the current process.\n"
            f"Support: Thanks for flagging this. Can you provide more details?\n"
            f"User: The issue occurs intermittently during execution.\n"
            f"Support: Acknowledged. We will investigate and follow up."
        )

    return f"{artifact_type} communication for {entity_name}."

def generate_decision_content(artifact_type: str, entity_name: str) -> str:
    decision_contexts = [
        "A review was conducted following a reported issue.",
        "An operational request required management review.",
        "An exception was raised during a standard workflow.",
        "A policy threshold was exceeded and required evaluation."
    ]

    decisions = {
        "Approval": "The request was approved to proceed as proposed.",
        "Escalation": "The issue was escalated to senior stakeholders for further review.",
        "Resolution": "The issue was resolved after corrective actions were applied."
    }

    return f"""
{artifact_type} — {entity_name}

Context
-------
{random.choice(decision_contexts)}

Decision
--------
{decisions.get(artifact_type, "A decision was recorded.")}

Rationale
---------
The decision was made based on risk assessment, operational impact, and compliance considerations.

Outcome
-------
Appropriate follow-up actions were initiated and tracked to completion.
""".strip()


def generate_content(category: str, artifact_type: str, entity_name: str) -> str:
    if category == "documents":
        return generate_document_content(artifact_type, entity_name)

    if category == "communications":
        return generate_communication_content(artifact_type, entity_name)

    if category == "processes":
        return (
            f"{artifact_type} — {entity_name}\n\n"
            f"Objective\n"
            f"---------\n"
            f"Define a repeatable process used by {entity_name} to ensure consistent execution, accountability, and timely escalation of issues.\n\n"
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
    if category == "structured_records":
        return generate_structured_record_content(artifact_type, entity_name)


    return (
        f"{artifact_type} for {entity_name}. "
        f"This {category[:-1]} defines internal guidance, context, and operational details "
        f"used by the organization."
    )

def generate_content(category: str, artifact_type: str, entity_name: str) -> str:
    if category == "documents":
        return generate_document_content(artifact_type, entity_name)

    if category == "communications":
        return generate_communication_content(artifact_type, entity_name)

    if category == "processes":
        return (
            f"{artifact_type} — {entity_name}\n\n"
            f"Objective\n"
            f"---------\n"
            f"Define a repeatable process used by {entity_name} to ensure consistent execution, accountability, and timely escalation of issues.\n\n"
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

    if category == "structured_records":
        return generate_structured_record_content(artifact_type, entity_name)

    if category == "decisions":
        return generate_decision_content(artifact_type, entity_name)

    return (
        f"{artifact_type} for {entity_name}. "
        f"This {category[:-1]} defines internal guidance, context, and operational details "
        f"used by the organization."
    )


def generate_artifact(entity_id: str, entity_name: str, category: str) -> dict:
    artifact_type = random.choice(CATEGORIES[category])
    return {
        "id": str(uuid.uuid4()),
        "entity_id": entity_id,
        "category": category,
        "artifact_type": artifact_type,
        "content": generate_content(category, artifact_type, entity_name),
        "governance": random_governance(),
    }


def derive_training_item(artifact: dict) -> dict:
    return {
        "entity_id": artifact["entity_id"],
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
    entitys: int,
    artifacts_per_category: int,
    output_dir: Path,
    derive_training: bool,
    training_ratio: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts_file = output_dir / "b2b_artifacts_v2.jsonl"
    training_file = output_dir / "b2b_training_items.jsonl"

    entitys_info = [
        (f"entity-{i:03d}", f"entityCorp{i:03d}") for i in range(1, entitys + 1)
    ]

    with open(artifacts_file, "w") as af, open(training_file, "w") if derive_training else open(artifacts_file, "a") as tf:
        for entity_id, entity_name in entitys_info:
            for category in CATEGORIES:
                for _ in range(artifacts_per_category):
                    artifact = generate_artifact(entity_id, entity_name, category)
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

# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate B2B synthetic enterprise data")
    parser.add_argument("--entitys", type=int, default=3)
    parser.add_argument("--artifacts-per-category", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--derive-training", action="store_true")
    parser.add_argument("--training-ratio", type=float, default=0.2)
    parser.add_argument("--purge-entity", type=str)

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
