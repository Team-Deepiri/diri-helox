#!/usr/bin/env python3
"""
Generate synthetic training data for task classification
31 categories: debugging, refactoring, writing_code, programming, running_code, inspecting,
writing, learning_research, learning_study, learning_training, learning_practice,
creative, administrative, team_organization, team_collaboration, team_planning,
research, planning, communication, big_data_analytics, data_processing, design,
qa, testing, validation, reporting, documentation, system_admin, ux_ui, security, data_privacy
Enhanced with Ollama integration for better synthetic data generation
"""
import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional
import sys
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import semantic analyzer for dynamic analysis
try:
    from app.train.utils.semantic_analyzer import get_semantic_analyzer
    HAS_SEMANTIC_ANALYZER = True
except ImportError:
    HAS_SEMANTIC_ANALYZER = False
    print("⚠ Semantic analyzer not available. Using template-based generation only.")

# Label mapping (31 categories)
LABEL_MAPPING = {
    # Coding breakdown (6 categories)
    "debugging": 0,
    "refactoring": 1,
    "writing_code": 2,
    "programming": 3,
    "running_code": 4,
    "inspecting": 5,
    # Core categories
    "writing": 6,
    "learning_research": 7,
    "learning_study": 8,
    "learning_training": 9,
    "learning_practice": 10,
    "creative": 11,
    "administrative": 12,
    # Team organization (3 categories)
    "team_organization": 13,
    "team_collaboration": 14,
    "team_planning": 15,
    # Computer/Desk-Work
    "research": 16,
    "planning": 17,
    "communication": 18,
    "big_data_analytics": 19,
    "data_processing": 20,
    "design": 21,
    "qa": 22,
    "testing": 23,
    "validation": 24,
    "reporting": 25,
    "documentation": 26,
    "system_admin": 27,
    # New categories
    "ux_ui": 28,
    "security": 29,
    "data_privacy": 30
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

# Common bare verbs that should not appear at the end of a sentence
BARE_VERBS = {
    "write", "create", "implement", "generate", "process", "review", 
    "run", "test", "validate", "inspect", "organize", "schedule", 
    "plan", "design", "debug", "fix", "troubleshoot", "develop",
    "build", "deploy", "configure", "setup", "install", "update",
    "analyze", "evaluate", "assess", "monitor", "track", "measure",
    "prepare", "collect", "gather", "compile", "document", "refactor"
}

def fix_bare_verb_at_end(text: str) -> str:
    """
    If the sentence ends with a bare verb, rewrite it into natural English.
    
    Examples:
      "A white paper on industry trends write" -> "Write a white paper on industry trends"
      "Inventory data process" -> "Process inventory data"
      "Team workflows organize" -> "Organize team workflows"
      "A design system create" -> "Create a design system"
    
    Args:
        text: The text to check and potentially fix
        
    Returns:
        The corrected text with proper verb-object order
    """
    if not text or not text.strip():
        return text
    
    # Strip and split into words
    text = text.strip()
    words = text.split()
    
    if len(words) < 2:
        return text
    
    # Get the last word (strip punctuation)
    last_word = words[-1].rstrip('.,!?;:').lower()
    
    # Check if it ends with a bare verb
    if last_word in BARE_VERBS:
        # Extract the verb and the rest of the phrase
        verb = words[-1].rstrip('.,!?;:')
        
        # Get punctuation if any
        punctuation = ''.join(c for c in words[-1] if c in '.,!?;:')
        
        # Get the object phrase (everything before the verb)
        object_phrase = ' '.join(words[:-1])
        
        # Remove leading articles if present (to avoid "a the system" issues)
        # and lowercase the object phrase for consistency
        object_phrase_lower = object_phrase.lower()
        if object_phrase_lower.startswith('a '):
            object_phrase = object_phrase_lower[2:]
        elif object_phrase_lower.startswith('an '):
            object_phrase = object_phrase_lower[3:]
        elif object_phrase_lower.startswith('the '):
            object_phrase = object_phrase_lower[4:]
        else:
            object_phrase = object_phrase_lower
        
        # Rebuild: Verb (capitalized) + object phrase + punctuation
        # Capitalize the verb
        verb_capitalized = verb.capitalize()
        
        # Construct the new sentence
        fixed_text = f"{verb_capitalized} {object_phrase}{punctuation}".strip()
        
        # Check if the verb already appears at the start of the object phrase
        # to avoid "Write write a document" situations
        object_words = object_phrase.split()
        if object_words and object_words[0] == verb.lower():
            # Already has verb at start, just return object phrase capitalized
            return object_phrase.capitalize() + punctuation
        
        return fixed_text
    
    return text

def is_valid_sentence(text: str) -> bool:
    """
    Validate that a sentence follows natural English order.
    Returns False if the sentence ends with a bare verb.
    
    Args:
        text: The text to validate
        
    Returns:
        True if the sentence is valid, False otherwise
    """
    if not text or not text.strip():
        return False
    
    # Get the last word (lowercase, stripped of punctuation)
    words = text.strip().rstrip('.!?').split()
    if not words:
        return False
    
    last_word = words[-1].lower().strip('.,!?;:')
    
    # Check if the last word is a bare verb
    if last_word in BARE_VERBS:
        return False
    
    return True

# Task templates for each category
TASK_TEMPLATES = {
    "debugging": [
        "Debug the payment processing API",
        "Fix the bug in the database connection pool",
        "Debug memory leaks in the application",
        "Troubleshoot the authentication middleware",
        "Fix the error in the search algorithm",
        "Debug the API endpoint timeout issue",
        "Fix the bug causing data corruption",
        "Debug the race condition in the service",
        "Troubleshoot the database query performance",
        "Fix the issue with file uploads",
        "Debug the caching layer problem",
        "Fix the bug in the error handling",
        "Troubleshoot the network connectivity issue",
        "Debug the session management problem",
        "Fix the issue with API rate limiting",
        "Debug the memory allocation error",
        "Troubleshoot the deployment failure",
        "Fix the bug in the validation logic",
        "Debug the integration test failures",
        "Fix the issue with the logging system",
        "Troubleshoot the performance degradation",
        "Debug the security vulnerability",
        "Fix the bug in the data migration",
        "Debug the issue with the API gateway",
        "Fix the problem with the message queue"
    ],
    "refactoring": [
        "Refactor the user service module",
        "Refactor legacy code to use modern patterns",
        "Refactor the authentication system",
        "Restructure the database access layer",
        "Refactor the API endpoints for clarity",
        "Improve code organization in the project",
        "Refactor the error handling mechanism",
        "Restructure the configuration management",
        "Refactor the data processing pipeline",
        "Improve the code structure in the module",
        "Refactor the caching implementation",
        "Restructure the service layer",
        "Refactor the validation logic",
        "Improve code readability in the component",
        "Refactor the logging system",
        "Restructure the API response format",
        "Refactor the database query methods",
        "Improve the architecture of the service",
        "Refactor the test suite structure",
        "Restructure the deployment scripts",
        "Refactor the monitoring implementation",
        "Improve code maintainability",
        "Refactor the integration layer",
        "Restructure the configuration files",
        "Refactor the utility functions"
    ],
    "writing_code": [
        "Write unit tests for my authentication API endpoints",
        "Write code to handle file uploads",
        "Write integration tests for the checkout flow",
        "Write unit tests for the validation logic",
        "Write a script to migrate data to new schema",
        "Write code for the new feature",
        "Write the API endpoint handler",
        "Write the database migration script",
        "Write code for error handling",
        "Write the authentication middleware",
        "Write code for the caching layer",
        "Write the data processing function",
        "Write code for the API client",
        "Write the configuration loader",
        "Write code for the logging system",
        "Write the utility functions",
        "Write code for the validation rules",
        "Write the service layer implementation",
        "Write code for the data model",
        "Write the integration test suite",
        "Write code for the API gateway",
        "Write the monitoring dashboard code",
        "Write code for the message queue",
        "Write the deployment automation script",
        "Write code for the security layer"
    ],
    "programming": [
        "Implement a new feature for user authentication",
        "Implement error handling for the API",
        "Create a new microservice for notifications",
        "Implement authentication middleware",
        "Implement rate limiting for API endpoints",
        "Program the data synchronization service",
        "Implement the caching strategy",
        "Create the API endpoint structure",
        "Implement the database connection pool",
        "Program the background job processor",
        "Implement the file processing system",
        "Create the message queue consumer",
        "Implement the API gateway routing",
        "Program the data transformation pipeline",
        "Implement the monitoring system",
        "Create the configuration management system",
        "Implement the logging framework",
        "Program the deployment automation",
        "Implement the security layer",
        "Create the integration layer",
        "Implement the validation framework",
        "Program the data access layer",
        "Implement the service orchestration",
        "Create the API documentation generator",
        "Implement the performance optimization"
    ],
    "running_code": [
        "Run the test suite for the project",
        "Execute the data migration script",
        "Run the integration tests",
        "Execute the deployment pipeline",
        "Run the performance benchmarks",
        "Execute the database backup script",
        "Run the code quality checks",
        "Execute the security scan",
        "Run the API endpoint tests",
        "Execute the data processing job",
        "Run the monitoring dashboard",
        "Execute the log analysis script",
        "Run the load testing suite",
        "Execute the database optimization",
        "Run the code coverage analysis",
        "Execute the dependency update",
        "Run the build process",
        "Execute the deployment validation",
        "Run the smoke tests",
        "Execute the data export script",
        "Run the performance profiling",
        "Execute the configuration validation",
        "Run the API documentation generator",
        "Execute the backup verification",
        "Run the system health checks"
    ],
    "inspecting": [
        "Review and optimize the search algorithm",
        "Inspect the codebase for security issues",
        "Review the API endpoint performance",
        "Inspect the database query patterns",
        "Review the error logs for patterns",
        "Inspect the system architecture",
        "Review the code quality metrics",
        "Inspect the API response times",
        "Review the database schema design",
        "Inspect the caching effectiveness",
        "Review the deployment configuration",
        "Inspect the monitoring dashboards",
        "Review the test coverage reports",
        "Inspect the dependency versions",
        "Review the security audit results",
        "Inspect the performance bottlenecks",
        "Review the code review comments",
        "Inspect the integration points",
        "Review the documentation completeness",
        "Inspect the error handling patterns",
        "Review the logging implementation",
        "Inspect the data flow architecture",
        "Review the scalability considerations",
        "Inspect the compliance requirements",
        "Review the system reliability metrics"
    ],
    "writing": [
        "Write a blog post about machine learning trends",
        "Draft an email to the team about the project update",
        "Create documentation for the new feature",
        "Write a technical report on the system architecture",
        "Compose a proposal for the client meeting",
        "Write a summary of the quarterly results",
        "Draft a press release for the product launch",
        "Create user guides for the application",
        "Write meeting notes from the sprint planning",
        "Compose a response to customer feedback",
        "Write a case study on the project success",
        "Draft a newsletter for the team",
        "Create content for the company website",
        "Write a research paper on the findings",
        "Compose a grant proposal for funding",
        "Write a tutorial on how to use the system",
        "Draft a contract for the new partnership",
        "Create marketing copy for the campaign",
        "Write a review of the latest technology",
        "Compose a letter to stakeholders",
        "Write documentation for API endpoints",
        "Draft a presentation script for the conference",
        "Create a style guide for the documentation",
        "Write a white paper on industry trends",
        "Compose a thank you note to the team"
    ],
    "learning_research": [
        "Read a research paper on transformers",
        "Research the latest industry trends",
        "Investigate new technology frameworks",
        "Read articles about best practices",
        "Research design patterns and architectures",
        "Investigate performance optimization techniques",
        "Read the latest industry research",
        "Research user experience best practices",
        "Investigate scalability solutions",
        "Read documentation for the library",
        "Research deployment strategies",
        "Investigate security best practices",
        "Read case studies on similar projects",
        "Research team collaboration tools",
        "Investigate monitoring and logging solutions",
        "Read technical specifications",
        "Research backup and disaster recovery",
        "Investigate cost optimization methods",
        "Read troubleshooting guides",
        "Research industry standards and compliance",
        "Investigate integration options",
        "Read academic papers on the topic",
        "Research market trends",
        "Investigate competitor solutions",
        "Read technical blog posts"
    ],
    "learning_study": [
        "Study for the certification exam",
        "Study the documentation for the API",
        "Study for the upcoming presentation",
        "Study the system design patterns",
        "Study the company's codebase",
        "Study for the technical interview",
        "Study the security best practices",
        "Read a chapter from the technical book",
        "Study the database schema",
        "Study the API architecture",
        "Study the deployment process",
        "Study the testing methodologies",
        "Study the code review process",
        "Study the project management approach",
        "Study the team collaboration methods",
        "Review study materials for the exam",
        "Study the system architecture",
        "Study the performance optimization",
        "Study the security protocols",
        "Study the data structures",
        "Study the algorithms",
        "Study the design principles",
        "Study the coding standards",
        "Study the documentation format",
        "Study the best practices"
    ],
    "learning_training": [
        "Take an online course on machine learning",
        "Attend a workshop on cloud architecture",
        "Take a course on data structures",
        "Attend training on the new framework",
        "Take a certification course",
        "Attend a technical workshop",
        "Take an online training program",
        "Attend a conference session",
        "Take a course on system design",
        "Attend training on security practices",
        "Take a programming bootcamp",
        "Attend a DevOps workshop",
        "Take a course on API design",
        "Attend training on database optimization",
        "Take an online certification program",
        "Attend a team training session",
        "Take a course on project management",
        "Attend training on agile methodologies",
        "Take a technical skills course",
        "Attend a leadership workshop",
        "Take a course on communication",
        "Attend training on new tools",
        "Take a course on best practices",
        "Attend a professional development session",
        "Take an advanced training program"
    ],
    "learning_practice": [
        "Practice coding problems on LeetCode",
        "Practice a new programming language",
        "Practice solving algorithm problems",
        "Practice with the new development tools",
        "Practice with the database queries",
        "Practice writing unit tests",
        "Practice code review techniques",
        "Practice debugging skills",
        "Practice refactoring code",
        "Practice API design",
        "Practice system architecture",
        "Practice deployment procedures",
        "Practice troubleshooting",
        "Practice performance optimization",
        "Practice security testing",
        "Practice writing documentation",
        "Practice team collaboration",
        "Practice presentation skills",
        "Practice technical writing",
        "Practice problem solving",
        "Practice design patterns",
        "Practice code organization",
        "Practice testing strategies",
        "Practice monitoring setup",
        "Practice automation scripting"
    ],
    "creative": [
        "Design a logo for the new project",
        "Write a short story for the contest",
        "Create a video montage of the trip",
        "Compose a piece of music",
        "Paint a landscape scene",
        "Write a poem about nature",
        "Create a digital art piece",
        "Design a poster for the event",
        "Write a screenplay for a short film",
        "Sketch ideas for the new product",
        "Create a photo collage",
        "Write a song with lyrics",
        "Create an animation for the project",
        "Design a brand identity package",
        "Write a creative blog post",
        "Create a video tutorial",
        "Write a children's story",
        "Create a portfolio website",
        "Write a script for a podcast",
        "Create a digital illustration",
        "Design a book cover",
        "Create a marketing campaign",
        "Write creative content",
        "Design visual assets",
        "Create multimedia presentations"
    ],
    "administrative": [
        "Schedule a meeting with the team",
        "File taxes for the year",
        "Pay bills and update budget",
        "Respond to important emails",
        "Update the project timeline",
        "Organize calendar for next week",
        "Review and approve expense reports",
        "Update employee records",
        "Schedule interviews for candidates",
        "Prepare agenda for the meeting",
        "File paperwork for the project",
        "Update the company policies",
        "Review contracts and agreements",
        "Organize files and documents",
        "Schedule appointments for the week",
        "Update the budget spreadsheet",
        "Respond to client inquiries",
        "Prepare reports for management",
        "Update the project status",
        "Organize team meeting notes",
        "File insurance claims",
        "Update contact information",
        "Schedule training sessions",
        "Review and process invoices",
        "Update the employee handbook"
    ],
    "team_organization": [
        "Organize a team building event",
        "Organize a group outing",
        "Organize a networking event",
        "Organize a study group",
        "Organize a community event",
        "Organize a volunteer activity",
        "Organize team meeting structure",
        "Organize the project team",
        "Organize team resources",
        "Organize team documentation",
        "Organize team workflows",
        "Organize team communication channels",
        "Organize team knowledge base",
        "Organize team training materials",
        "Organize team code repositories",
        "Organize team project files",
        "Organize team meeting schedules",
        "Organize team responsibilities",
        "Organize team performance reviews",
        "Organize team feedback sessions",
        "Organize team retrospectives",
        "Organize team celebrations",
        "Organize team onboarding process",
        "Organize team offboarding process",
        "Organize team recognition program"
    ],
    "team_collaboration": [
        "Reach out to a colleague for coffee",
        "Reach out to an old friend",
        "Reach out to mentors",
        "Collaborate on the project design",
        "Work together on the feature",
        "Pair program with a teammate",
        "Collaborate on code review",
        "Work on the team project",
        "Collaborate on documentation",
        "Work together on testing",
        "Collaborate on problem solving",
        "Work on team improvements",
        "Collaborate on architecture decisions",
        "Work together on deployment",
        "Collaborate on troubleshooting",
        "Work on team knowledge sharing",
        "Collaborate on best practices",
        "Work together on optimization",
        "Collaborate on security review",
        "Work on team communication",
        "Collaborate on planning",
        "Work together on execution",
        "Collaborate on evaluation",
        "Work on team feedback",
        "Collaborate on continuous improvement"
    ],
    "team_planning": [
        "Plan a team lunch",
        "Plan a birthday celebration",
        "Plan a weekend trip with friends",
        "Plan a surprise party",
        "Plan a family gathering",
        "Plan the team sprint",
        "Plan team capacity allocation",
        "Plan team skill development",
        "Plan team project milestones",
        "Plan team resource needs",
        "Plan team communication strategy",
        "Plan team meeting agenda",
        "Plan team retrospective format",
        "Plan team training schedule",
        "Plan team performance goals",
        "Plan team collaboration approach",
        "Plan team workflow improvements",
        "Plan team technology adoption",
        "Plan team process optimization",
        "Plan team knowledge management",
        "Plan team documentation structure",
        "Plan team code review process",
        "Plan team deployment strategy",
        "Plan team monitoring approach",
        "Plan team continuous improvement"
    ],
    "research": [
        "Research market trends for the new product",
        "Investigate competitors in the industry",
        "Look up best practices for API design",
        "Research user behavior patterns",
        "Find information about the latest technology",
        "Investigate pricing strategies",
        "Research customer feedback and reviews",
        "Look up technical specifications",
        "Investigate security vulnerabilities",
        "Research industry standards and compliance",
        "Find case studies on similar projects",
        "Investigate performance optimization techniques",
        "Research design patterns and architectures",
        "Look up documentation for the framework",
        "Investigate integration options",
        "Research user experience best practices",
        "Find statistics and data sources",
        "Investigate scalability solutions",
        "Research deployment strategies",
        "Look up troubleshooting guides",
        "Investigate cost optimization methods",
        "Research team collaboration tools",
        "Find information about regulations",
        "Investigate monitoring and logging solutions",
        "Research backup and disaster recovery"
    ],
    "planning": [
        "Plan the project roadmap for Q1",
        "Create a sprint plan for the team",
        "Plan the architecture for the new system",
        "Schedule milestones for the project",
        "Plan resource allocation for next month",
        "Create a timeline for the feature release",
        "Plan the budget for the quarter",
        "Schedule team meetings for the week",
        "Plan the testing strategy",
        "Create a deployment plan",
        "Plan the migration strategy",
        "Schedule code review sessions",
        "Plan the training schedule",
        "Create a risk mitigation plan",
        "Plan the communication strategy",
        "Schedule stakeholder meetings",
        "Plan the documentation structure",
        "Create a contingency plan",
        "Plan the performance benchmarks",
        "Schedule release dates",
        "Plan the security audit",
        "Create a maintenance schedule",
        "Plan the feature prioritization",
        "Schedule team retrospectives",
        "Plan the capacity expansion"
    ],
    "communication": [
        "Send an email to the team about the update",
        "Schedule a meeting with stakeholders",
        "Write a status update for the project",
        "Respond to client inquiries",
        "Send a follow-up message to the team",
        "Schedule a one-on-one with my manager",
        "Write a progress report",
        "Send meeting invitations",
        "Respond to urgent emails",
        "Schedule a conference call",
        "Write a project update message",
        "Send feedback to team members",
        "Schedule a demo presentation",
        "Respond to customer support tickets",
        "Write a team announcement",
        "Send a reminder about the deadline",
        "Schedule a training session",
        "Write a proposal for the client",
        "Send a thank you message",
        "Schedule a brainstorming session",
        "Respond to vendor inquiries",
        "Write a meeting summary",
        "Send a status notification",
        "Schedule a review meeting",
        "Write a change request"
    ],
    "big_data_analytics": [
        "Analyze large-scale user engagement data",
        "Process petabytes of transaction data",
        "Create analytics on massive datasets",
        "Analyze big data for customer insights",
        "Process streaming data analytics",
        "Create big data visualizations",
        "Analyze distributed system metrics",
        "Process real-time analytics data",
        "Create predictive models on big data",
        "Analyze data warehouse information",
        "Process cloud-scale analytics",
        "Create machine learning analytics",
        "Analyze time-series big data",
        "Process distributed data analytics",
        "Create big data dashboards",
        "Analyze social media big data",
        "Process IoT sensor data analytics",
        "Create big data trend analysis",
        "Analyze multi-source data integration",
        "Process big data for business intelligence",
        "Create big data correlation analysis",
        "Analyze data lake contents",
        "Process big data for pattern recognition",
        "Create big data performance metrics",
        "Analyze distributed analytics results"
    ],
    "data_processing": [
        "Process sales data for the quarter",
        "Process financial data for the month",
        "Process survey responses",
        "Process log files for errors",
        "Process transaction records",
        "Process customer feedback data",
        "Process inventory data",
        "Process time series data",
        "Process API response data",
        "Process database export files",
        "Process CSV files for analysis",
        "Process JSON data structures",
        "Process image data for training",
        "Process text data for NLP",
        "Process sensor data streams",
        "Process email data for analysis",
        "Process configuration files",
        "Process backup data files",
        "Process migration data",
        "Process validation data",
        "Process transformation data",
        "Process aggregation data",
        "Process normalization data",
        "Process enrichment data",
        "Process cleaning data pipelines"
    ],
    "design": [
        "Design a user interface mockup",
        "Create wireframes for the new feature",
        "Design a logo for the project",
        "Create a color palette for the brand",
        "Design a dashboard layout",
        "Create icon designs for the app",
        "Design a landing page",
        "Create a style guide",
        "Design a mobile app interface",
        "Create a presentation template",
        "Design a form layout",
        "Create a visual identity",
        "Design a navigation structure",
        "Create a design system",
        "Design a user flow diagram",
        "Create a prototype",
        "Design a data visualization",
        "Create a marketing banner",
        "Design a email template",
        "Create a infographic",
        "Design a report layout",
        "Create a social media graphic",
        "Design a packaging concept",
        "Create a animation storyboard",
        "Design a website layout"
    ],
    "qa": [
        "Perform quality assurance testing",
        "Conduct QA review of the feature",
        "Execute QA test cases",
        "Perform QA regression testing",
        "Conduct QA smoke testing",
        "Execute QA integration tests",
        "Perform QA user acceptance testing",
        "Conduct QA performance testing",
        "Execute QA security testing",
        "Perform QA accessibility testing",
        "Conduct QA compatibility testing",
        "Execute QA load testing",
        "Perform QA stress testing",
        "Conduct QA usability testing",
        "Execute QA exploratory testing",
        "Perform QA boundary testing",
        "Conduct QA negative testing",
        "Execute QA positive testing",
        "Perform QA cross-browser testing",
        "Conduct QA mobile device testing",
        "Execute QA API testing",
        "Perform QA database testing",
        "Conduct QA end-to-end testing",
        "Execute QA automated testing",
        "Perform QA manual testing"
    ],
    "testing": [
        "Write unit tests for the new feature",
        "Run integration tests for the API",
        "Perform regression testing",
        "Test the user interface components",
        "Run performance tests",
        "Test the authentication flow",
        "Perform security testing",
        "Test the payment processing",
        "Run load tests on the server",
        "Test the mobile app on devices",
        "Perform accessibility testing",
        "Test the error handling",
        "Run end-to-end tests",
        "Test the data migration script",
        "Perform usability testing",
        "Test the API endpoints",
        "Run smoke tests",
        "Test the deployment process",
        "Perform compatibility testing",
        "Test the backup and restore",
        "Run stress tests",
        "Test the notification system",
        "Perform penetration testing",
        "Test the search functionality",
        "Run automated test suite"
    ],
    "validation": [
        "Validate the user input data",
        "Validate the API request format",
        "Validate the database schema",
        "Validate the configuration settings",
        "Validate the security credentials",
        "Validate the data integrity",
        "Validate the business rules",
        "Validate the system requirements",
        "Validate the deployment configuration",
        "Validate the integration points",
        "Validate the data transformation",
        "Validate the error handling",
        "Validate the performance metrics",
        "Validate the compliance requirements",
        "Validate the accessibility standards",
        "Validate the code quality",
        "Validate the documentation",
        "Validate the test coverage",
        "Validate the security measures",
        "Validate the backup procedures",
        "Validate the monitoring setup",
        "Validate the logging implementation",
        "Validate the data privacy",
        "Validate the user permissions",
        "Validate the system reliability"
    ],
    "reporting": [
        "Create a report on website traffic",
        "Generate a quarterly performance report",
        "Create a data quality report",
        "Generate a test coverage report",
        "Create a security audit report",
        "Generate a deployment status report",
        "Create a user engagement report",
        "Generate a system health report",
        "Create a cost analysis report",
        "Generate a compliance report",
        "Create a bug tracking report",
        "Generate a performance metrics report",
        "Create a team productivity report",
        "Generate a project status report",
        "Create a customer feedback report",
        "Generate a financial summary report",
        "Create a risk assessment report",
        "Generate a quality metrics report",
        "Create a training completion report",
        "Generate a incident response report",
        "Create a capacity planning report",
        "Generate a trend analysis report",
        "Create a benchmark comparison report",
        "Generate a recommendations report",
        "Create a executive summary report"
    ],
    "documentation": [
        "Write API documentation for the endpoints",
        "Create a user guide for the feature",
        "Document the architecture decisions",
        "Write installation instructions",
        "Create a troubleshooting guide",
        "Document the database schema",
        "Write a developer onboarding guide",
        "Create a changelog for the release",
        "Document the deployment process",
        "Write a configuration guide",
        "Create a FAQ document",
        "Document the API integration",
        "Write a security policy",
        "Create a runbook for operations",
        "Document the testing procedures",
        "Write a migration guide",
        "Create a style guide",
        "Document the code standards",
        "Write a performance tuning guide",
        "Create a disaster recovery plan",
        "Document the monitoring setup",
        "Write a contribution guide",
        "Create a glossary of terms",
        "Document the backup procedures",
        "Write a release notes document"
    ],
    "system_admin": [
        "Set up a new server environment",
        "Configure the database backup",
        "Update system security patches",
        "Monitor server performance",
        "Configure the load balancer",
        "Set up monitoring and alerts",
        "Update SSL certificates",
        "Configure firewall rules",
        "Set up automated backups",
        "Monitor disk space usage",
        "Configure the DNS settings",
        "Update system dependencies",
        "Set up log rotation",
        "Configure the reverse proxy",
        "Update the operating system",
        "Set up user access controls",
        "Configure the email server",
        "Monitor network traffic",
        "Set up the staging environment",
        "Configure the CI/CD pipeline",
        "Update the database version",
        "Set up the development environment",
        "Configure the caching layer",
        "Monitor application logs",
        "Set up the production environment"
    ],
    "ux_ui": [
        "Design the user experience flow",
        "Create user interface mockups",
        "Design the interaction patterns",
        "Create wireframes for the feature",
        "Design the visual hierarchy",
        "Create user journey maps",
        "Design the information architecture",
        "Create prototype interactions",
        "Design the responsive layout",
        "Create user interface components",
        "Design the navigation system",
        "Create accessibility features",
        "Design the color scheme",
        "Create typography guidelines",
        "Design the icon system",
        "Create animation guidelines",
        "Design the micro-interactions",
        "Create user testing scenarios",
        "Design the onboarding experience",
        "Create usability guidelines",
        "Design the error states",
        "Create loading state designs",
        "Design the empty states",
        "Create feedback mechanisms",
        "Design the mobile interface"
    ],
    "security": [
        "Perform security audit of the system",
        "Implement security best practices",
        "Review security vulnerabilities",
        "Configure security policies",
        "Implement authentication security",
        "Review access control mechanisms",
        "Implement encryption for data",
        "Configure security monitoring",
        "Review security compliance",
        "Implement secure coding practices",
        "Review security incident response",
        "Configure security firewalls",
        "Implement security logging",
        "Review security architecture",
        "Implement security testing",
        "Configure security certificates",
        "Review security documentation",
        "Implement security training",
        "Configure security alerts",
        "Review security risk assessment",
        "Implement security patches",
        "Configure security backups",
        "Review security access logs",
        "Implement security protocols",
        "Configure security automation"
    ],
    "data_privacy": [
        "Review data privacy policies",
        "Implement GDPR compliance measures",
        "Audit data privacy practices",
        "Configure data privacy settings",
        "Review data collection practices",
        "Implement data anonymization",
        "Audit data access controls",
        "Configure data retention policies",
        "Review data sharing agreements",
        "Implement data encryption",
        "Audit data processing procedures",
        "Configure privacy consent management",
        "Review data subject rights",
        "Implement data deletion procedures",
        "Audit third-party data sharing",
        "Configure privacy notifications",
        "Review data breach procedures",
        "Implement privacy by design",
        "Audit data privacy documentation",
        "Configure privacy impact assessments",
        "Review data privacy training",
        "Implement privacy monitoring",
        "Audit data privacy compliance",
        "Configure privacy reporting",
        "Review data privacy governance"
    ]
}

def generate_variations(
    base_text: str, 
    category: str, 
    num_variations: int = 3,
    use_ollama: bool = False,
    semantic_analyzer = None
) -> List[str]:
    """
    Generate variations of a base task text with dynamic semantic analysis
    Uses CMU-inspired semantic analysis approaches for dynamic variation generation
    
    Args:
        base_text: Base task text
        category: Task category
        num_variations: Number of variations to generate
        use_ollama: Whether to use Ollama for semantic analysis
        semantic_analyzer: SemanticAnalyzer instance if available
    """
    variations = [base_text]  # Include original
    
    # Use dynamic semantic analysis if available (optimized to minimize Ollama calls)
    if use_ollama and semantic_analyzer and HAS_SEMANTIC_ANALYZER:
        try:
            # Get cached prefixes and suffixes (cached per category, not per text)
            # This avoids repeated Ollama calls for the same category
            prefixes = semantic_analyzer.generate_semantic_prefixes(base_text, category)
            suffixes = semantic_analyzer.generate_semantic_suffixes(base_text, category)
            
            # Generate template-based variations with dynamic prefixes/suffixes
            template_variations = []
            for prefix in prefixes[:min(3, num_variations - 1)]:
                for suffix in suffixes[:1]:  # Use first suffix only for templates
                    variation = f"{prefix} {base_text.lower()}{suffix}".strip()
                    if variation != base_text and variation not in template_variations and is_valid_sentence(variation):
                        template_variations.append(variation)
            
            variations.extend(template_variations[:num_variations - 1])
            
            # Only use expensive Ollama calls (paraphrases, verbs) if we still need more variations
            # Skip if we already have enough
            if len(variations) < num_variations:
                # Generate semantic paraphrases using Ollama (only if needed)
                try:
                    paraphrases = semantic_analyzer.generate_paraphrases(
                        base_text, 
                        category, 
                        min(2, num_variations - len(variations))  # Limit to 2 paraphrases max
                    )
                    # Validate paraphrases before adding
                    for paraphrase in paraphrases:
                        if is_valid_sentence(paraphrase):
                            variations.append(paraphrase)
                except Exception as e:
                    pass  # Skip paraphrases if Ollama fails, continue with what we have
            
            # Use semantic verb extraction for additional variations (cached per category)
            if len(variations) < num_variations:
                semantic_verbs = semantic_analyzer.extract_semantic_verbs(base_text, category)
                if semantic_verbs:
                    words = base_text.split()
                    if len(words) > 1:
                        # Replace first word (likely verb) with semantic alternatives
                        for verb in semantic_verbs[:min(3, num_variations - len(variations))]:
                            if verb.lower() != words[0].lower():
                                new_text = f"{verb} {' '.join(words[1:])}"
                                if new_text not in variations and is_valid_sentence(new_text):
                                    variations.append(new_text)
                                    if len(variations) >= num_variations:
                                        break
            
        except Exception as e:
            # Silently fall back to default templates (don't spam errors)
            use_ollama = False
    
    # Fallback: Use default templates if semantic analysis not available
    if not use_ollama or len(variations) < num_variations:
        # Default prefixes (fallback)
        default_prefixes = [
            "I need to",
            "Can you help me",
            "Please",
            "I want to",
            "Help me",
            "I should",
            "Let me",
            "I'm going to"
        ]
        
        # Default suffixes (fallback)
        default_suffixes = [
            "",
            " today",
            " this week",
            " as soon as possible",
            " when you have time",
            " - urgent",
            " - important"
        ]
        
        # Generate template-based variations
        template_variations = []
        for prefix in default_prefixes[:min(3, num_variations - len(variations))]:
            for suffix in default_suffixes[:1]:
                variation = f"{prefix} {base_text.lower()}{suffix}".strip()
                if variation != base_text and variation not in variations and is_valid_sentence(variation):
                    template_variations.append(variation)
        
        variations.extend(template_variations)
    
    # Ensure we have enough variations (with safety limit to prevent infinite loop)
    attempts = 0
    max_attempts = 10
    while len(variations) < num_variations and attempts < max_attempts:
        attempts += 1
        # Generate additional variations using simple transformations
        # that maintain proper sentence structure
        words = base_text.split()
        
        # Try adding contextual modifiers to the beginning (keeping verb structure intact)
        if len(words) >= 2:
            modifiers = ["Quickly", "Carefully", "Thoroughly", "Efficiently"]
            for modifier in modifiers:
                # Add modifier after the first word (verb) to maintain structure
                # e.g., "Review the code" -> "Carefully review the code"
                if words[0][0].isupper():  # If it starts with a capital (likely a verb)
                    new_text = f"{modifier} {base_text.lower()}".strip()
                else:
                    new_text = f"{modifier} {base_text}".strip()
                
                if new_text not in variations and is_valid_sentence(new_text):
                    variations.append(new_text)
                    if len(variations) >= num_variations:
                        break
            
            if len(variations) >= num_variations:
                break
        
        # If still need more, just break instead of creating malformed sentences
        break
    
    # If still not enough, pad with the base text (better than malformed sentences)
    while len(variations) < num_variations:
        variations.append(base_text)
    
    return variations[:num_variations]

def generate_synthetic_dataset(
    total_examples: int = 7000,  # Increased default
    examples_per_class: int = None,
    output_dir: str = "app/train/data",
    use_ollama: bool = False  # Disabled by default for speed (enable with --use-ollama)
) -> Dict:
    """
    Generate synthetic dataset for task classification
    
    Args:
        total_examples: Total number of examples to generate
        examples_per_class: If specified, generate this many per class (overrides total_examples)
        output_dir: Directory to save the dataset
        use_ollama: Whether to use Ollama for enhanced data generation
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate examples per class
    if examples_per_class:
        total_examples = examples_per_class * len(LABEL_MAPPING)
    
    examples_per_class = total_examples // len(LABEL_MAPPING)
    remainder = total_examples % len(LABEL_MAPPING)
    
    print("=" * 60)
    print("Generating Synthetic Training Data")
    print("=" * 60)
    print(f"Total examples: {total_examples}")
    print(f"Examples per class: ~{examples_per_class}")
    print(f"Categories: {len(LABEL_MAPPING)} total")
    print(f"Semantic augmentation: {'Enabled (slower)' if use_ollama else 'Disabled (fast, template-based)'}")
    if use_ollama:
        print(f"⚠️  Note: Ollama will be used selectively (every 20th template) for speed")
    print()
    
    # Initialize semantic analyzer if requested
    semantic_analyzer = None
    if use_ollama and HAS_SEMANTIC_ANALYZER:
        try:
            print("Initializing semantic analyzer (Ollama-based)...")
            semantic_analyzer = get_semantic_analyzer()
            if semantic_analyzer and semantic_analyzer.check_ollama_available():
                print("✓ Semantic analyzer initialized successfully")
                print(f"  Using Ollama at: {semantic_analyzer.ollama_base_url}")
                print(f"  Model: {semantic_analyzer.model}")
            else:
                print("⚠ Ollama not available, falling back to template-based generation")
                semantic_analyzer = None
        except Exception as e:
            print(f"⚠ Failed to initialize semantic analyzer: {e}")
            print("   Continuing with template-based generation only")
            semantic_analyzer = None
    
    all_data = []
    label_counts = Counter()
    
    # Generate data for each category
    total_categories = len(LABEL_MAPPING)
    for cat_idx, (category, label_id) in enumerate(LABEL_MAPPING.items(), 1):
        templates = TASK_TEMPLATES[category]
        num_examples = examples_per_class + (1 if label_id < remainder else 0)
        
        print(f"[{cat_idx}/{total_categories}] Generating {num_examples} examples for '{category}'...")
        
        # Calculate how many variations per template
        variations_per_template = max(1, num_examples // len(templates))
        extra_variations = num_examples % len(templates)
        
        category_data = []
        # Pre-cache semantic prefixes/suffixes for this category (once per category, not per template)
        if use_ollama and semantic_analyzer is not None:
            try:
                # Pre-cache category-level data to avoid repeated Ollama calls
                sample_template = templates[0] if templates else ""
                print(f"    Pre-caching semantic data for '{category}'...")
                _ = semantic_analyzer.generate_semantic_prefixes(sample_template, category)
                print(f"      ✓ Prefixes cached")
                _ = semantic_analyzer.generate_semantic_suffixes(sample_template, category)
                print(f"      ✓ Suffixes cached")
                _ = semantic_analyzer.extract_semantic_verbs(sample_template, category)
                print(f"      ✓ Verbs cached")
            except Exception as e:
                print(f"      ⚠ Pre-caching failed: {e}")
                pass  # Continue even if pre-caching fails
        
        for i, template in enumerate(templates):
            num_variations = variations_per_template + (1 if i < extra_variations else 0)
            # Use semantic analysis much more selectively: only every 20th template (was every 5th)
            # This reduces Ollama calls by 75%
            use_semantic_for_this = use_ollama and (i % 20 == 0) and semantic_analyzer is not None
            variations = generate_variations(
                template, 
                category, 
                num_variations,
                use_ollama=use_semantic_for_this,
                semantic_analyzer=semantic_analyzer
            )
            
            # Progress indicator every 5 templates
            if (i + 1) % 5 == 0:
                print(f"    Progress: {i + 1}/{len(templates)} templates processed ({len(category_data)} examples so far)")
            
            for variation in variations:
                if len(category_data) >= num_examples:
                    break
                
                # Apply post-processing to fix any bare verbs at the end
                # This transforms malformed text like "Inventory data process" 
                # into "Process inventory data"
                fixed_text = fix_bare_verb_at_end(variation)
                
                # Final validation: skip truly invalid sentences
                if not is_valid_sentence(fixed_text):
                    continue
                
                task_id = f"task_{len(all_data) + len(category_data):06d}"
                
                example = {
                    "id": task_id,
                    "text": fixed_text,
                    "label": category,
                    "label_id": label_id,
                    "metadata": {
                        "length": len(fixed_text),
                        "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                        "source": "synthetic"
                    }
                }
                
                category_data.append(example)
                label_counts[category] += 1
        
        all_data.extend(category_data)
        print(f"  ✓ Generated {len(category_data)} examples for '{category}'")
        if use_ollama and semantic_analyzer:
            cache_size = len(semantic_analyzer._cache)
            print(f"    (Semantic cache: {cache_size} entries)")
    
    # Shuffle the data
    random.shuffle(all_data)
    
    # Split into train (70%), validation (15%), test (15%)
    total = len(all_data)
    train_size = int(total * 0.70)
    val_size = int(total * 0.15)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # Save datasets
    train_file = output_path / "classification_train.jsonl"
    val_file = output_path / "classification_val.jsonl"
    test_file = output_path / "classification_test.jsonl"
    
    print(f"\nSaving datasets...")
    print(f"  Train: {len(train_data)} examples -> {train_file}")
    print(f"  Val: {len(val_data)} examples -> {val_file}")
    print(f"  Test: {len(test_data)} examples -> {test_file}")
    
    # Save in format expected by training script (text, label as integer)
    for file_path, data in [(train_file, train_data), (val_file, val_data), (test_file, test_data)]:
        with open(file_path, 'w') as f:
            for item in data:
                # Save in format expected by trainer: {"text": "...", "label": 0}
                f.write(json.dumps({
                    "text": item["text"],
                    "label": item["label_id"]
                }) + '\n')
    
    # Also save full format with metadata
    full_train_file = output_path / "synthetic_classification_train.jsonl"
    full_val_file = output_path / "synthetic_classification_val.jsonl"
    full_test_file = output_path / "synthetic_classification_test.jsonl"
    
    for file_path, data in [(full_train_file, train_data), (full_val_file, val_data), (full_test_file, test_data)]:
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    # Save label mapping
    label_map_file = output_path / "label_mapping.json"
    with open(label_map_file, 'w') as f:
        json.dump({
            "label2id": LABEL_MAPPING,
            "id2label": ID_TO_LABEL
        }, f, indent=2)
    
    # Save metadata
    metadata = {
        "dataset_name": "deepiri-task-classification-v1",
        "version": "1.0",
        "total_samples": total,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "num_classes": len(LABEL_MAPPING),
        "label_distribution": dict(label_counts),
        "avg_text_length": sum(len(item["text"]) for item in all_data) / len(all_data),
        "min_text_length": min(len(item["text"]) for item in all_data),
        "max_text_length": max(len(item["text"]) for item in all_data)
    }
    
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {total}")
    print(f"  Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation: {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/total*100:.1f}%)")
    print(f"\nLabel Distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} examples")
    print(f"\nFiles saved:")
    print(f"  Training data: {train_file}")
    print(f"  Validation data: {val_file}")
    print(f"  Test data: {test_file}")
    print(f"  Label mapping: {label_map_file}")
    print(f"  Metadata: {metadata_file}")
    print(f"\nNext step: Run training")
    print(f"  python app/train/scripts/train_intent_classifier.py")
    
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "metadata": metadata
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument(
        "--total-examples",
        type=int,
        default=7000,
        help="Total number of examples to generate (default: 7000)"
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=None,
        help="Number of examples per class (overrides total-examples)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="app/train/data",
        help="Output directory for datasets (default: app/train/data)"
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        default=False,
        help="Use Ollama for enhanced data augmentation (slower, default: False for speed)"
    )
    parser.add_argument(
        "--no-ollama",
        dest="use_ollama",
        action="store_false",
        help="Disable Ollama augmentation"
    )
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        total_examples=args.total_examples,
        examples_per_class=args.examples_per_class,
        output_dir=args.output_dir,
        use_ollama=args.use_ollama
    )

