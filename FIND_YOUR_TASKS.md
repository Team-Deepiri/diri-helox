# FIND YOUR TASKS - Deepiri Team Guide

**Quick Navigation:** Use Ctrl+F to find your role



---

### Founding Engineer
**Reports to:** Founder
**Location:** `deepiri/` (Platform-wide)
**Stack:** System Architecture, AI Research, Scalability Planning
**Start Here:**
- Architecture docs: `deepiri/docs/SYSTEM_ARCHITECTURE.md`
- Microservices: `deepiri/docs/MICROSERVICES_SETUP.md`
- AI research: `deepiri/python_backend/train/`
**Your Tasks:**
- Technical vision
- Platform architecture
- AI research direction
- Scalability planning
- Team leadership

---

### Product Lead
**Reports to:** Founder
**Location:** `deepiri/` (Product-wide)
**Stack:** Product Management, User Research, UX Design
**Start Here:**
- Product checklist: `deepiri/docs/PRODUCT_CHECKLIST.md`
- Frontend: `deepiri/frontend/`
- User flows: `deepiri/docs/`
**Your Tasks:**
- User research
- Product-market fit
- Feature prioritization
- UX design for gamification
- Roadmap alignment

---

## AI TEAM

### AI Research Lead (AI Scientist 0)
**Reports to:** Founder
**Location:** `deepiri/python_backend/train/`
**Stack:** Python, PyTorch, Transformers, MLflow, W&B, Jupyter
**Start Here:**
- Review: `deepiri/python_backend/train/README.md`
- Experiment tracking: `deepiri/python_backend/train/infrastructure/experiment_tracker.py`
- Research templates: `deepiri/python_backend/train/experiments/research_experiment_template.py`
**Unique Mission:** Design conceptual models that define Deepiri's cognitive engine and gamification logic. Lead theoretical development of Deepiri's cognition model.
**Your Tasks:**
- Define personalization frameworks for adaptive challenge generation
- Design challenge generation theory that maps tasks → gamified challenges
- Explore model architectures conceptually (paper-level research)
- Provide research direction to engineering teams
- Maintain AI design blueprint document (`python_backend/train/README.md`)
- Oversee LLM development and model orchestration
- Coordinate cutting-edge AI integration (RAG, RL, multi-agent systems)
- Guide fine-tuning strategies for task classification and challenge generation
- Review challenge generation algorithms and cognitive load balancing models
- Establish research priorities for multimodal understanding and model compression

---

### AI Research Scientist 1 - Cognitive Task Structuring
**Reports to:** AI Research Lead & AI Systems Lead
**Location:** `deepiri/python_backend/train/experiments/`
**Stack:** Python, PyTorch, Transformers, Novel Architectures (Mamba, MoE), Custom Training Loops, Graph Neural Networks
**Start Here:**
- Template: `deepiri/python_backend/train/experiments/research_experiment_template.py`
- Notebooks: `deepiri/python_backend/train/notebooks/`
- Neuro-symbolic service: `deepiri/python_backend/app/services/neuro_symbolic_challenge.py`
**Unique Mission:** Theorize how tasks become "challenges" in the single-player gamified system. Build theoretical frameworks for task-to-challenge conversion.
**Your Tasks:**
- Research cognitive load balancing algorithms for challenge difficulty
- Study decision-making frameworks for daily planning and task prioritization
- Explore non-transformer cognitive models (Mamba, MoE, GNN-level theory)
- Create mathematical models for difficulty scaling based on user performance
- Research novel neural architectures for gamification logic
- Develop new training methodologies for task understanding (`task_classifier.py`)
- Experiment with Mamba and MoE architectures for task processing
- Research alternative optimization algorithms beyond Adam/AdamW
- Support ML Engineers with theoretical foundations (not systems implementation)
**Files to Create:**
- `deepiri/python_backend/train/experiments/mamba_architecture.py`
- `deepiri/python_backend/train/experiments/moe_gamification.py`
- `deepiri/python_backend/train/experiments/neuro_symbolic_experiments.py`
- `deepiri/python_backend/train/experiments/cognitive_load_balancing.py`
- `deepiri/python_backend/train/experiments/difficulty_scaling_models.py`
- `deepiri/python_backend/train/experiments/decision_making_frameworks.py`

---

### AI Research Scientist 2 - Multimodal Understanding Theory
**Reports to:** AI Research Lead & AI Systems Lead
**Location:** `deepiri/python_backend/train/experiments/multimodal/`
**Stack:** Python, PyTorch, Multimodal Models (CLIP, BLIP), Vision Transformers, Audio Processing, Semantic Graphs, Symbolic AI
**Start Here:**
- Multimodal service: `deepiri/python_backend/app/services/multimodal_understanding.py`
- Graph neural networks: `deepiri/python_backend/train/experiments/gnn_task_relationships.py`
**Unique Mission:** Theorize how text → visuals → actions → code → audio fuse into a unified representation for task understanding and challenge generation.
**Your Tasks:**
- Research theoretical multimodal fusion for task parsing (`multimodal_understanding.py`)
- Study conceptual semantic graphs for knowledge representation
- Explore symbolic + deep hybrid models (neuro-symbolic AI)
- Provide theoretical groundwork for creative task gamification
- Create conceptual visual reasoning strategies
- Design frameworks for document parsing (PDFs, images, code repos)
- Research graph neural networks for task relationship modeling
- Support ML Engineers with multimodal research (not systems implementation)
**Files to Create:**
- `deepiri/python_backend/train/experiments/multimodal_fusion.py`
- `deepiri/python_backend/train/experiments/visual_reasoning.py`
- `deepiri/python_backend/train/experiments/graph_neural_networks.py`
- `deepiri/python_backend/train/experiments/semantic_graphs.py`
- `deepiri/python_backend/train/experiments/symbolic_deep_hybrid.py`
- `deepiri/python_backend/train/experiments/unified_representation.py`

---

### AI Research Scientist 3 - Efficiency & Model Compression Theory
**Reports to:** AI Research Lead & AI Systems Lead
**Location:** `deepiri/python_backend/train/experiments/compression/`
**Stack:** Python, PyTorch, Quantization (GPTQ, QLoRA), Pruning, Distillation, ONNX, TensorRT, Federated Learning
**Start Here:**
- LoRA training: `deepiri/python_backend/train/infrastructure/lora_training.py`
- Model compression experiments
- Federated learning: `deepiri/python_backend/train/experiments/federated_learning.py`
**Unique Mission:** Design efficient model architectures for on-device inference and cloud optimization.
**Your Tasks:**
- Study optimal quantization strategies (4-bit/8-bit with bitsandbytes)
- Explore sparse network theory for model compression
- Theorize architectures suitable for on-device inference (desktop IDE)
- Investigate federated learning frameworks theoretically
- Propose conceptual memory-efficient training paradigms
- Research LoRA adapter optimization for per-user personalization
- Design model distillation strategies for smaller, faster models
- Support AI Systems for native app inference optimization
**Files to Create:**
- `deepiri/python_backend/train/experiments/quantization_methods.py`
- `deepiri/python_backend/train/experiments/pruning_techniques.py`
- `deepiri/python_backend/train/experiments/sparse_training.py`
- `deepiri/python_backend/train/experiments/federated_learning.py`
- `deepiri/python_backend/train/experiments/optimal_quantization.py`
- `deepiri/python_backend/train/experiments/sparse_network_theory.py`
- `deepiri/python_backend/train/experiments/on_device_architectures.py`
- `deepiri/python_backend/train/experiments/memory_efficient_training.py`

---

### AI Systems Lead
**Reports to:** Founder
**Location:** `deepiri/python_backend/app/services/` & `deepiri/python_backend/train/pipelines/`
**Stack:** Python, FastAPI, Docker, Kubernetes, MLflow, Model Deployment
**Start Here:**
- Services: `deepiri/python_backend/app/services/`
- Training pipelines: `deepiri/python_backend/train/pipelines/`
- Main app: `deepiri/python_backend/app/main.py`
**Unique Mission:** Model deployment infrastructure, inference optimization, training pipeline management. Code review and merging for AI Systems team.
**Your Tasks:**
- Manage model deployment infrastructure for production
- Optimize inference pipelines for latency and cost
- Oversee training pipeline management (MLflow, W&B integration)
- Review and merge code from AI Systems Engineers
- Coordinate with ML Team Lead on model deployment strategy
- Manage GPU cluster scheduling and resource allocation
- Ensure production-grade model serving (`inference_service.py`)
- Maintain model versioning and deployment automation
- Coordinate with MLOps Engineers on CI/CD pipelines

---

### AI Systems Engineer 1 - Model Inference Routing
**Reports to:** AI Systems Lead
**Location:** `deepiri/python_backend/app/services/` & `deepiri/python_backend/app/routes/`
**Stack:** Python, FastAPI, OpenAI API, Anthropic API, AsyncIO, WebSockets, Model Orchestration, Routing Logic
**Start Here:**
- Inference service: `deepiri/python_backend/app/services/inference_service.py`
- Hybrid AI: `deepiri/python_backend/app/services/hybrid_ai_service.py`
- Challenge routes: `deepiri/python_backend/app/routes/challenge.py`
- Task routes: `deepiri/python_backend/app/routes/task.py`
- Agent routes: `deepiri/python_backend/app/routes/agent.py`
**Unique Mission:** Develop inference-routing logic for both web and native app platforms.
**Your Tasks:**
- Build orchestration logic for model selection (`model_selector.py`)
- Optimize prompt processing pipelines for FastAPI endpoints
- Route requests between task-agent ↔ plan-agent ↔ code-agent
- Reduce latency in challenge generation API (`/api/challenges/generate`)
- Design fallback models for weak connectivity scenarios
- Implement caching strategies for repeated queries
- Optimize batch processing for concurrent requests
- Monitor inference performance and latency metrics
**Files to Work On:**
- `deepiri/python_backend/app/services/inference_service.py` - Inference routing
- `deepiri/python_backend/app/services/model_selector.py` - Model selection logic
- `deepiri/python_backend/app/services/agent_routing.py` - Agent routing (create)
- `deepiri/python_backend/app/routes/challenge.py` - API routes
- `deepiri/python_backend/app/routes/task.py` - Task routes
- `deepiri/python_backend/app/routes/agent.py` - Agent routes
**Files to Create:**
- `deepiri/python_backend/app/services/inference_router.py` - Main routing service
- `deepiri/python_backend/app/services/fallback_models.py` - Fallback logic
- `deepiri/python_backend/app/services/prompt_pipeline.py` - Prompt optimization

---

### AI Systems Engineer 2 - Agent Interaction Framework
**Reports to:** AI Systems Lead
**Location:** `deepiri/python_backend/app/services/`
**Stack:** Python, FastAPI, AsyncIO, Message Queues, Multi-Agent Systems, Safety Frameworks
**Start Here:**
- Multi-agent system: `deepiri/python_backend/app/services/multi_agent_system.py`
- Context-aware adaptation: `deepiri/python_backend/app/services/context_aware_adaptation.py`
- PPO agent: `deepiri/python_backend/app/services/ppo_agent.py`
**Unique Mission:** Build and maintain internal messaging framework for multi-agent reasoning.
**Your Tasks:**
- Architect agent → agent communication layers (`multi_agent_system.py`)
- Build reasoning loop frameworks for collaborative AI agents
- Add safety + guardrails + alignment mechanisms
- Support model orchestration during complex tasks
- Implement agent coordination protocols
- Design error handling and recovery for agent failures
- Build agent state management and context sharing
- Ensure agent outputs align with challenge generation requirements
**Files to Work On:**
- `deepiri/python_backend/app/services/multi_agent_system.py` - Multi-agent coordination
- `deepiri/python_backend/app/services/context_aware_adaptation.py` - Context adaptation
- `deepiri/python_backend/app/services/ppo_agent.py` - PPO agent
**Files to Create:**
- `deepiri/python_backend/app/services/agent_messaging.py` - Agent communication framework
- `deepiri/python_backend/app/services/reasoning_loops.py` - Reasoning loop frameworks
- `deepiri/python_backend/app/services/safety_guardrails.py` - Safety mechanisms
- `deepiri/python_backend/app/services/alignment_mechanisms.py` - Alignment systems

---

### AI Systems Engineer 3 - Distributed Training Infrastructure
**Reports to:** AI Systems Lead
**Location:** `deepiri/python_backend/train/pipelines/` & `deepiri/python_backend/mlops/`
**Stack:** Python, PyTorch, Distributed Training (DeepSpeed, Ray, Slurm), Kubernetes, MLflow, GPU Clusters
**Start Here:**
- Training pipeline: `deepiri/python_backend/train/pipelines/full_training_pipeline.py`
- Distributed training: `deepiri/python_backend/train/pipelines/distributed_training.py`
- MLOps: `deepiri/python_backend/mlops/`
**Unique Mission:** Own cloud-side scaling + distributed model training.
**Your Tasks:**
- Build Ray/K8s/Slurm-style distributed training infrastructure
- Manage GPU cluster scheduling (RTX 4090s, H100s)
- Optimize multiprocessing for training pipelines
- Implement data parallelism and model parallelism (DeepSpeed ZeRO)
- Configure distributed training for LoRA adapters
- Manage training job queues and resource allocation
- Optimize training data loading and preprocessing
- Coordinate with MLOps Engineers on training automation
**Files to Work On:**
- `deepiri/python_backend/train/pipelines/distributed_training.py`
- `deepiri/python_backend/mlops/ci/training_pipeline.yml`
- `deepiri/python_backend/mlops/deployment/model_deployment.py` (create)
**Files to Create:**
- `deepiri/python_backend/train/pipelines/ray_distributed.py` - Ray integration
- `deepiri/python_backend/train/pipelines/slurm_distributed.py` - Slurm integration
- `deepiri/python_backend/train/pipelines/gpu_scheduler.py` - GPU scheduling
- `deepiri/python_backend/train/pipelines/data_parallel.py` - Data parallelism
- `deepiri/python_backend/train/pipelines/model_parallel.py` - Model parallelism

---

### AI Systems Engineer 4 - Production-Grade Model Serving
**Reports to:** AI Systems Lead
**Location:** `deepiri/python_backend/mlops/deployment/` & `deepiri/python_backend/app/services/`
**Stack:** Python, FastAPI, Model Serving (Triton, vLLM), Kubernetes, Batching, Concurrency, Versioning
**Start Here:**
- Inference service: `deepiri/python_backend/app/services/inference_service.py`
- MLOps deployment: `deepiri/python_backend/mlops/deployment/deployment_automation.py`
- Model registry: `deepiri/python_backend/mlops/registry/model_registry.py`
**Unique Mission:** Build resilient serving infrastructure for both web and desktop IDE platforms.
**Your Tasks:**
- Production inference servers (FastAPI + Uvicorn)
- Build scalable model-serving endpoints (`/api/challenges/*`, `/api/tasks/*`)
- Optimize batching + concurrency for high-throughput
- Handle versioned deployment across web and desktop platforms
- Implement model A/B testing infrastructure
- Build model health monitoring and auto-scaling
- Optimize GPU memory usage for concurrent requests
- Work closely with MLOps Engineers on deployment pipelines
**Files to Work On:**
- `deepiri/python_backend/app/services/inference_service.py` - Inference serving
- `deepiri/python_backend/mlops/deployment/deployment_automation.py` - Deployment
- `deepiri/python_backend/mlops/registry/model_registry.py` - Model registry
**Files to Create:**
- `deepiri/python_backend/app/services/model_serving.py` - Production serving
- `deepiri/python_backend/app/services/batching_optimizer.py` - Batching logic
- `deepiri/python_backend/app/services/versioned_deployment.py` - Versioning
- `deepiri/python_backend/mlops/deployment/triton_serving.py` - Triton integration
- `deepiri/python_backend/mlops/deployment/vllm_serving.py` - vLLM integration

---

### ML Engineer 0 (ML Team Lead)
**Reports to:** Founder
**Location:** `deepiri/python_backend/train/` & `deepiri/python_backend/mlops/`
**Stack:** Python, PyTorch, ML Strategy, Model Architecture, Cross-team Coordination
**Start Here:**
- Training README: `deepiri/python_backend/train/README.md`
- ML config: `deepiri/python_backend/app/train/configs/ml_training_config.json`
- MLOps README: `deepiri/python_backend/mlops/README.md`
**Unique Mission:** Orchestrate ML strategy, technical direction, and cross-team alignment. Does NOT merge code (AI Systems Lead reviews and merges).
**Your Tasks:**
- Technical architecture and model strategy for Deepiri's AI systems
- Resource allocation and project prioritization across ML team
- Cross-functional collaboration with product/engineering teams
- ML system design and scalability planning
- Team mentorship and technical guidance
- Model deployment governance and quality standards
- Coordinate with AI Research Lead on research priorities
- Align ML roadmap with product roadmap
- Review model performance metrics and optimization opportunities

---

### ML Engineer 1 — Gamification & Reinforcement Learning
**Reports to:** ML Team Lead
**Location:** `deepiri/python_backend/train/scripts/` & `deepiri/python_backend/app/services/`
**Stack:** Python, PyTorch, Reinforcement Learning (RLlib, Stable-Baselines3), Multi-Armed Bandits, Actor-Critic
**Start Here:**
- Bandit training: `deepiri/python_backend/train/pipelines/bandit_training.py`
- Bandit service: `deepiri/python_backend/app/services/bandit_service.py`
- PPO agent: `deepiri/python_backend/app/services/ppo_agent.py`
- RL environment: `deepiri/python_backend/app/services/rl_environment.py`
- Reward model: `deepiri/python_backend/app/services/reward_model.py`
**Unique Mission:** Turn tasks → challenges using RL frameworks and reward modeling.
**Your Tasks:**
- Train reward models (`reward_model.py`) for challenge generation
- Build challenge selection policies using RL algorithms
- Actor-critic optimization for adaptive difficulty (`ppo_agent.py`)
- Personalized difficulty engines based on user performance
- Implement RL environment for challenge generation (`rl_environment.py`)
- Train bandit algorithms for challenge recommendation (`bandit_service.py`)
- Optimize reward functions for user engagement
- Integrate RL models with challenge generation pipeline
**Files to Create:**
- `deepiri/python_backend/train/scripts/train_policy_network.py`
- `deepiri/python_backend/train/scripts/train_value_network.py`
- `deepiri/python_backend/train/scripts/train_actor_critic.py`
- `deepiri/python_backend/train/scripts/train_reward_model.py`
- `deepiri/python_backend/train/scripts/train_challenge_selection.py`
- `deepiri/python_backend/train/scripts/train_difficulty_engine.py`

---

### ML Engineer 2 - Transformers, Distillation, Task Understanding
**Reports to:** ML Team Lead
**Location:** `deepiri/python_backend/train/scripts/` & `deepiri/python_backend/app/services/`
**Stack:** Python, PyTorch, Transformers, PEFT/LoRA, Quantization (bitsandbytes, GPTQ), Knowledge Distillation
**Start Here:**
- Task classifier training: `deepiri/python_backend/train/scripts/train_task_classifier.py`
- Transformer training: `deepiri/python_backend/train/scripts/train_transformer_classifier.py`
- LoRA training: `deepiri/python_backend/train/infrastructure/lora_training.py`
- Task classifier service: `deepiri/python_backend/app/services/task_classifier.py`
- Advanced task parser: `deepiri/python_backend/app/services/advanced_task_parser.py`
**Unique Mission:** Train and optimize transformer models for task understanding and classification.
**Your Tasks:**
- Train task classification transformers (`task_classifier.py`)
- Create agent-teacher-student systems for model distillation
- Distill large models into smaller ones for faster inference
- Build efficient variants for on-device deployment
- Fine-tune DeBERTa-v3 for task classification
- Optimize transformer models for challenge generation
- Implement knowledge distillation pipelines
- Train models for task parsing and understanding
**Files to Work On:**
- `deepiri/python_backend/train/scripts/train_task_classifier.py`
- `deepiri/python_backend/train/scripts/train_transformer_classifier.py`
- `deepiri/python_backend/train/scripts/train_teacher_student.py` (create)
- `deepiri/python_backend/train/scripts/train_quantized_model.py` (create)
**Files to Create:**
- `deepiri/python_backend/train/scripts/train_task_understanding.py` - Deep task understanding
- `deepiri/python_backend/train/scripts/train_knowledge_distillation.py` - Distillation
- `deepiri/python_backend/train/scripts/train_edge_efficient.py` - Edge models

---

### ML Engineer 3 - Behavior & Temporal Learning
**Reports to:** ML Team Lead
**Location:** `deepiri/python_backend/train/scripts/` & `deepiri/python_backend/app/services/`
**Stack:** Python, PyTorch, Lightweight Models (MobileNet, EfficientNet), Temporal Models (LSTM, GRU, Transformers), Ensemble Methods
**Start Here:**
- Challenge generator training: `deepiri/python_backend/train/scripts/train_challenge_generator.py`
- Personalization training: `deepiri/python_backend/train/scripts/train_personalization_model.py`
- Session analyzer: `deepiri/python_backend/app/services/session_analyzer.py`
**Unique Mission:** Model user habits and predict behavior patterns for personalization.
**Your Tasks:**
- Train temporal sequence models for user behavior prediction
- Build habit-prediction networks for challenge timing
- Train ensemble scoring models for user performance
- Local recommendation models for challenge suggestions
- Implement time-series analysis for productivity patterns
- Build user behavior clustering models
- Predict optimal challenge timing based on user history
- Integrate behavior models with gamification service
**Files to Create:**
- `deepiri/python_backend/train/scripts/train_lightweight_challenge_generator.py`
- `deepiri/python_backend/train/scripts/train_temporal_behavior_model.py`
- `deepiri/python_backend/train/scripts/train_ensemble_scoring.py`
- `deepiri/python_backend/train/scripts/train_on_device_recommendation.py`
- `deepiri/python_backend/train/scripts/train_temporal_sequence.py` - Temporal sequences
- `deepiri/python_backend/train/scripts/train_habit_prediction.py` - Habit prediction

---

### MLOps Engineer 1 - CI/CD for Models
**Reports to:** ML Team Lead
**Location:** `deepiri/python_backend/mlops/ci/` & `deepiri/python_backend/mlops/deployment/`
**Stack:** Python, CI/CD (GitHub Actions, GitLab CI), Kubernetes, MLflow, Docker, Prometheus, GPU Management
**Start Here:**
- **Onboarding Guide**: `deepiri/docs/MLOPS_TEAM_ONBOARDING.md` (READ THIS FIRST!)
- **CI/CD Pipeline**: `deepiri/python_backend/mlops/ci/model_ci_pipeline.py`
- **Deployment**: `deepiri/python_backend/mlops/deployment/deployment_automation.py`
- **Model Registry**: `deepiri/python_backend/mlops/registry/model_registry.py`
- **MLOps README**: `deepiri/python_backend/mlops/README.md`
- **Setup Script**: `deepiri/python_backend/mlops/scripts/setup_mlops_environment.sh`
**Unique Mission:** Automate model build → test → deploy pipelines.
**Your Tasks:**
- Model versioning with MLflow model registry
- Training automation for scheduled retraining
- GPU resource lifecycle management
- Build reproducibility for training experiments
- CI/CD pipeline configuration (`mlops/ci/training_pipeline.yml`)
- Automated model testing and validation
- Model artifact management and storage
- Integration with Kubernetes for training jobs
**Files to Work On:**
- `deepiri/python_backend/mlops/ci/model_ci_pipeline.py` - Main CI/CD pipeline
- `deepiri/python_backend/mlops/deployment/deployment_automation.py` - Deployment strategies
- `deepiri/python_backend/mlops/registry/model_registry.py` - Model registry
- `deepiri/python_backend/mlops/scripts/run_ci_pipeline.sh` - CI/CD automation
- `deepiri/python_backend/mlops/scripts/deploy_model.sh` - Deployment script
**Files to Create:**
- `deepiri/python_backend/mlops/ci/github_actions.yml` - GitHub Actions workflow
- `deepiri/python_backend/mlops/deployment/kubernetes_deployment.yaml` - K8s manifests
- `deepiri/python_backend/mlops/gpu/gpu_manager.py` - GPU resource management

---

### MLOps Engineer 2 - Monitoring & Optimization
**Reports to:** ML Team Lead
**Location:** `deepiri/python_backend/mlops/monitoring/` & `deepiri/python_backend/mlops/optimization/`
**Stack:** Python, Prometheus, Grafana, MLflow, Performance Profiling, Alerting, Cost Optimization
**Start Here:**
- **Onboarding Guide**: `deepiri/docs/MLOPS_TEAM_ONBOARDING.md` (READ THIS FIRST!)
- **Model Monitoring**: `deepiri/python_backend/mlops/monitoring/model_monitor.py`
- **Docker Setup**: `deepiri/python_backend/mlops/docker/docker-compose.mlops.yml`
- **MLOps README**: `deepiri/python_backend/mlops/README.md`
**Unique Mission:** Create monitoring + performance insights for AI systems.
**Your Tasks:**
- Monitor model drift and performance degradation
- Track infrastructure LATENCY for inference endpoints
- Create alerting dashboards for model health
- Optimize AI costs (GPU usage, API calls, inference time)
- Performance monitoring for challenge generation API
- Model A/B testing infrastructure
- Resource utilization tracking and optimization
- Integration with Prometheus and Grafana for metrics
**Files to Work On:**
- `deepiri/python_backend/mlops/monitoring/model_monitor.py` - Main monitoring service
- `deepiri/python_backend/mlops/scripts/monitor_model.sh` - Monitoring script
- `deepiri/python_backend/mlops/docker/docker-compose.mlops.yml` - Monitoring stack
**Files to Create:**
- `deepiri/python_backend/mlops/monitoring/dashboards/` - Grafana dashboard configs
- `deepiri/python_backend/mlops/monitoring/alerts/` - Alert configurations
- `deepiri/python_backend/mlops/monitoring/drift_detection.py` - Advanced drift detection
- `deepiri/python_backend/mlops/monitoring/performance_metrics.py` - Performance tracking
- `deepiri/python_backend/mlops/optimization/inference_optimization.py` - Inference optimization
- `deepiri/python_backend/mlops/optimization/resource_optimizer.py` - Resource optimization

---

### Data Engineer 1 - Real-Time User Behavior Pipelines
**Reports to:** ML Team Lead
**Location:** `deepiri/python_backend/train/data/` & `deepiri/python_backend/app/services/analytics/`
**Stack:** Python, Pandas, NumPy, Apache Kafka, NATS, Real-time Processing, Feature Engineering, Event Streaming
**Start Here:**
- Data collection: `deepiri/python_backend/train/pipelines/data_collection_pipeline.py`
- Dataset prep: `deepiri/python_backend/train/data/prepare_dataset.py`
**Unique Mission:** Create real-time event pipelines to feed ML and Gamification services.
**Your Tasks:**
- Stream user activity via Kafka/NATS (or Redis Streams)
- Build challenge analytics pipelines
- Generate real-time features for ML models
- Integrate with Analytics Service for time-series data
- Real-time event processing for gamification triggers
- Data pipeline for user behavior tracking
- Event streaming for challenge completion events
- Integration with InfluxDB for time-series storage
**Files to Create:**
- `deepiri/python_backend/train/data/user_behavior_pipeline.py`
- `deepiri/python_backend/train/data/challenge_analytics_pipeline.py`
- `deepiri/python_backend/app/services/analytics/real_time_features.py`
- `deepiri/python_backend/app/services/analytics/event_processor.py`
- `deepiri/python_backend/app/services/analytics/kafka_streaming.py` - Kafka integration
- `deepiri/python_backend/app/services/analytics/nats_streaming.py` - NATS integration
- `deepiri/python_backend/app/services/analytics/challenge_analytics.py` - Challenge analytics

---

### Data Engineer 2 - Quality, Privacy, Compliance
**Reports to:** ML Team Lead
**Location:** `deepiri/python_backend/train/data/` (Data Quality & Privacy)
**Stack:** Python, Pandas, Data Validation, Privacy Tools (Differential Privacy, PII Detection), GDPR Compliance
**Start Here:**
- Dataset prep: `deepiri/python_backend/train/data/prepare_dataset.py`
- Data collection: `deepiri/python_backend/train/pipelines/data_collection_pipeline.py`
**Unique Mission:** Maintain complete data integrity & privacy constraints.
**Your Tasks:**
- Label validation for training datasets
- GDPR and data minimization compliance
- Data anonymization for user behavior data
- Dataset curation for model training
- Data quality assurance and validation
- Privacy-preserving data processing
- Data retention policy enforcement
- Compliance with data protection regulations
**Files to Create:**
- `deepiri/python_backend/train/data/data_curation.py`
- `deepiri/python_backend/train/data/data_quality.py`
- `deepiri/python_backend/train/data/privacy_anonymization.py`
- `deepiri/python_backend/train/data/pii_detection.py`
- `deepiri/python_backend/train/data/data_validation.py`
- `deepiri/python_backend/train/data/label_validation.py` - Label validation
- `deepiri/python_backend/train/data/gdpr_compliance.py` - GDPR compliance
- `deepiri/python_backend/train/data/data_minimization.py` - Data minimization

---

### AI Systems Intern 1
**Reports to:** AI Systems Lead
**Location:** `deepiri/python_backend/train/scripts/` & `deepiri/python_backend/docs/`
**Stack:** Python, PyTorch, Documentation Tools, Testing Frameworks
**Start Here:**
- Test suite: `deepiri/python_backend/tests/ai/`
- Training scripts: `deepiri/python_backend/train/scripts/`
- Documentation: `deepiri/docs/`
**Unique:** High-level documentation + assisting live training jobs.
**Your Tasks:**
- Support model training
- Write test cases for AI services
- Documentation for training scripts
- Test data preparation
- High-level documentation
- Assist with live training jobs
**Files to Work On:**
- `deepiri/python_backend/tests/ai/test_task_classifier.py`
- `deepiri/python_backend/tests/ai/test_challenge_generator.py`
- `deepiri/python_backend/train/scripts/README.md` (create/update)
- `deepiri/docs/ai_training_guide.md` (create/update)

---

### AI Systems Intern 2
**Reports to:** AI Systems Lead
**Location:** `desktop-ide-deepiri/src-tauri/src/` (Desktop/Edge AI)
**Stack:** Rust, ONNX Runtime, Quantized Models, Desktop Deployment, Testing
**Start Here:**
- Local LLM: `desktop-ide-deepiri/src-tauri/src/local_llm.rs`
- Tauri backend: `desktop-ide-deepiri/src-tauri/src/main.rs`
**Unique Mission:** Tests native desktop agent behavior on local environments.
**Your Tasks:**
- Test desktop IDE AI agent functionality
- Validate local model inference on desktop
- Test offline-first AI capabilities
- Validate desktop-to-cloud sync for AI features
- Report bugs and issues with desktop AI integration
**Files to Work On:**
- `desktop-ide-deepiri/src-tauri/src/local_llm.rs`
- `desktop-ide-deepiri/src-tauri/src/commands.rs`
- `desktop-ide-deepiri/tests/edge_ai_tests.rs` (create)
- `desktop-ide-deepiri/tests/native_agent_tests.rs` (create)

---

### AI Systems Intern 3 (ML Engineer Intern)
**Reports to:** AI Systems Lead
**Location:** `deepiri/python_backend/train/data/` & `deepiri/python_backend/train/scripts/`
**Stack:** Python, Pandas, Data Processing, ETL Pipelines, Synthetic Data Generation, Automation
**Start Here:**
- Dataset prep: `deepiri/python_backend/train/data/prepare_dataset.py`
- Data collection: `deepiri/python_backend/train/pipelines/data_collection_pipeline.py`
- Training scripts: `deepiri/python_backend/train/scripts/`
**Unique Mission:** Builds synthetic datagen + automation for training.
**Your Tasks:**
- Build synthetic data generation for training datasets
- Automate data collection and preprocessing
- Create data augmentation pipelines
- Generate synthetic challenges for training
- Automate dataset preparation workflows
**Files to Work On:**
- `deepiri/python_backend/train/data/prepare_dataset.py`
- `deepiri/python_backend/train/data/data_cleaning.py` (create)
- `deepiri/python_backend/train/data/etl_pipeline.py` (create)
**Files to Create:**
- `deepiri/python_backend/train/data/synthetic_data_generator.py` - Synthetic data
- `deepiri/python_backend/train/scripts/training_automation.py` - Training automation

---

### AI Systems Intern 4
**Reports to:** AI Systems Lead
**Location:** `deepiri/python_backend/tests/ai/benchmarks/` & `deepiri/python_backend/train/scripts/`
**Stack:** Python, Pytest, Benchmarking Tools, Performance Profiling, Regression Testing
**Start Here:**
- Benchmarks: `deepiri/python_backend/tests/ai/benchmarks/benchmark_classifier.py`
- Evaluation: `deepiri/python_backend/train/scripts/evaluate_model.py`
**Unique Mission:** Focuses on evaluating failure modes + regression tests.
**Your Tasks:**
- Evaluate AI system failure modes
- Build regression tests for model updates
- Test edge cases in challenge generation
- Validate model outputs for correctness
- Create test suites for AI services
**Files to Work On:**
- `deepiri/python_backend/tests/ai/benchmarks/benchmark_classifier.py`
- `deepiri/python_backend/train/scripts/evaluate_model.py`
- `deepiri/python_backend/tests/ai/benchmarks/benchmark_generator.py` (create)
- `deepiri/python_backend/train/scripts/quality_metrics.py` (create)
**Files to Create:**
- `deepiri/python_backend/tests/ai/failure_mode_analysis.py` - Failure mode evaluation
- `deepiri/python_backend/tests/ai/regression_tests.py` - Regression test suite

---

### AI Systems Intern 5
**Reports to:** AI Systems Lead
**Location:** `deepiri/python_backend/docs/` & `deepiri/services/shared-utils/`
**Stack:** Python, Documentation Tools, Testing, Code Quality, Refactoring
**Start Here:**
- Tests: `deepiri/python_backend/tests/`
- READMEs: `deepiri/python_backend/train/README.md`
- Shared utils: `deepiri/services/shared-utils/`
**Unique Mission:** Maintains shared-utils consistency + code refactors.
**Your Tasks:**
- Maintain shared utilities across AI services
- Refactor code for consistency and maintainability
- Ensure code quality standards across AI codebase
- Help with code organization and structure
- Support code review and refactoring efforts
**Files to Create:**
- `deepiri/python_backend/docs/ai_services.md`
- `deepiri/python_backend/docs/training_guide.md`
- `deepiri/python_backend/tests/README.md`
- `deepiri/python_backend/.pre-commit-config.yaml` (create)
- `deepiri/services/shared-utils/refactoring_guide.md` (create)

---

### AI Systems Intern 6
**Reports to:** AI Systems Lead
**Location:** `deepiri/python_backend/tests/ai/` & `deepiri/python_backend/train/experiments/`
**Stack:** Python, Pytest, Simulation Frameworks, Validation Tools, QA Testing
**Start Here:**
- AI tests: `deepiri/python_backend/tests/ai/`
- Integration tests: `deepiri/python_backend/tests/integration/`
**Unique Mission:** QA Agent testing, simulation environments, validation.
**Your Tasks:**
- QA testing for AI agents and services
- Build simulation environments for testing
- Validate agent behavior and outputs
- Test multi-agent system interactions
- Create test scenarios for challenge generation
**Files to Create:**
- `deepiri/python_backend/tests/ai/test_qa_agent.py`
- `deepiri/python_backend/train/experiments/simulation_env.py`
- `deepiri/python_backend/tests/integration/test_ai_pipeline.py`
- `deepiri/python_backend/tests/ai/simulation_environments.py` - Simulation environments
- `deepiri/python_backend/tests/ai/validation_scripts.py` - Validation scripts

---

## MICROSERVICES / BACKEND TEAM

### Backend Lead
**Reports to:** Product Lead / Founder
**Location:** `deepiri/services/` & `deepiri/python_backend/app/`
**Stack:** Node.js, Express, Python, FastAPI, MongoDB, Redis, Microservices Architecture
**Start Here:**
- Architecture: `deepiri/docs/MICROSERVICES_SETUP.md` - **READ THIS FIRST!**
- API Gateway: `deepiri/services/api-gateway/server.js` (Port 5000)
- Services: `deepiri/services/` - Each service has its own `server.js`
- Python backend: `deepiri/python_backend/app/main.py` (Port 8000)
- Docker Compose: `deepiri/docker-compose.dev.yml`
**Unique Mission:** Oversees microservices architecture, REST/GraphQL APIs, database schema quality, cross-team coordination, service boundaries, backend correctness. Reviews and merges code for backend team.
**Your Tasks:**
- Coordinate microservices architecture (9+ services)
- API strategy for REST endpoints and WebSocket connections
- Database patterns (MongoDB, Redis, InfluxDB)
- Cross-collaboration with AI Systems and Platform engineers
- Ensure web microservices and native/local IDE backend sync layer alignment
- Review and merge code from Backend Engineers
- Establish coding standards and best practices
- Coordinate service boundaries and API contracts
- Ensure backend correctness and reliability
**Key Files:**
- `deepiri/services/api-gateway/server.js` - API Gateway (routes all requests)
- `deepiri/services/*/server.js` - Individual service servers
- `deepiri/docker-compose.dev.yml` - Service orchestration

---

### Backend Engineer 1 - External Integrations
**Reports to:** Backend Lead
**Location:** `deepiri/services/integration-service/`
**Stack:** Node.js, Express, OAuth2, REST APIs, Webhooks, External API Integration
**Start Here:**
- Service server: `deepiri/services/integration-service/server.js` (Port 5006)
- Route handlers: `deepiri/services/integration-service/src/index.js`
- Webhook service: `deepiri/services/integration-service/src/webhookService.js`
- Integration service: `deepiri/services/integration-service/README.md`
**Unique Mission:** External Integrations: Notion/Trello/GitHub APIs, OAuth flows, webhook management, data synchronization.
**Your Tasks:**
- Build OAuth flows for Notion, Trello, GitHub, Google Docs
- Implement webhook management for external service updates
- Data synchronization between external services and Deepiri
- Authenticate all integration endpoints (JWT middleware)
- Ensure frontend, backend, and agent services have real-time communication
- Build integration service endpoints (`integration-service/`)
- Handle OAuth token refresh and management
- Parse and transform external service data formats
**Files to Work On:**
- `deepiri/services/integration-service/server.js` - Service server
- `deepiri/services/integration-service/src/index.js` - Route handlers
- `deepiri/services/integration-service/src/webhookService.js` - Webhook processing
- `deepiri/services/integration-service/Dockerfile` - Container definition
**Files to Create:**
- `deepiri/services/integration-service/src/notion.js` (if needed)
- `deepiri/services/integration-service/src/trello.js` (if needed)
- `deepiri/services/integration-service/src/github.js` (if needed)

---

### Backend Engineer 2 - WebSocket Infrastructure
**Reports to:** Backend Lead
**Location:** `deepiri/services/websocket-service/` & `deepiri/services/notification-service/`
**Stack:** Node.js, Socket.IO, WebSockets, Real-time Systems, Redis Pub/Sub
**Start Here:**
- WebSocket service: `deepiri/services/websocket-service/server.js` (Port 5008)
- Notification service: `deepiri/services/notification-service/server.js` (Port 5005)
- WebSocket handler: `deepiri/services/notification-service/src/websocketService.js`
**Unique Mission:** WebSocket infrastructure, message routing, presence tracking, active challenge sessions support.
**Your Tasks:**
- WebSocket infrastructure for real-time communication (`websocket-service/`)
- Message routing for live sessions
- Presence tracking for active users
- Active challenge sessions support
- Ensures low latency streams for cloud IDE interactions
- Syncs multiplayer state with AI orchestration layer for live "pair programming"
- Real-time challenge tracking and updates
- WebSocket connection management and scaling
**Files to Work On:**
- `deepiri/services/websocket-service/server.js` - WebSocket service
- `deepiri/services/notification-service/server.js` - Notification service
- `deepiri/services/notification-service/src/websocketService.js` - WebSocket handler
- `deepiri/services/websocket-service/Dockerfile` - Container definition

---

### Backend Engineer 3 - AI Integration & Gamification
**Reports to:** Backend Lead
**Location:** `deepiri/python_backend/app/routes/` & `deepiri/services/challenge-service/`
**Stack:** Python, FastAPI, Node.js, Express, AI Integration, State Management
**Start Here:**
- Challenge service: `deepiri/services/challenge-service/server.js` (Port 5007)
- Challenge routes: `deepiri/python_backend/app/routes/challenge.py` (Port 8000)
- AI services: `deepiri/python_backend/app/services/`
**Unique Mission:** AI Integration: Python service communication layer, challenge state management, gamification rule engine logic.
**Your Tasks:**
- Python service communication layer (FastAPI integration)
- Challenge state management and lifecycle
- Gamification rule engine logic + challenge state lifecycle
- AI validation pipeline for challenge generation
- AI-provided instructions mapping into tasks for both native IDE and online IDE
- Manages cross-context data structures (task states, user states, challenge metadata)
- Integrate with Python AI service (`http://pyagent:8000`)
- Challenge completion tracking and reward distribution
**Files to Work On:**
- `deepiri/services/challenge-service/server.js` - Challenge service (port 5007) - Main service file
- `deepiri/python_backend/app/routes/challenge.py` - Python AI routes
- `deepiri/services/challenge-service/Dockerfile` - Container definition
**Files to Create:**
- `deepiri/services/challenge-service/src/` directory (if needed for route handlers)
- `deepiri/services/challenge-service/src/challenge_state.js` (if needed)
- `deepiri/services/challenge-service/src/gamification_rules.js` (if needed)
- `deepiri/services/challenge-service/src/ai_validator.js` (if needed)

---

### Backend Engineer 4 - Database & Caching
**Reports to:** Backend Lead
**Location:** `deepiri/services/*/src/` & `deepiri/python_backend/app/database/`
**Stack:** MongoDB, Redis, InfluxDB, SQL, Database Optimization, Query Performance, Data Migrations
**Start Here:**
- User Service: `deepiri/services/user-service/server.js` (Port 5001) - Time-series tracking
- Analytics Service: `deepiri/services/analytics-service/server.js` (Port 5004) - InfluxDB
- Database models: `deepiri/python_backend/app/database/models.py`
- Database config: `deepiri/python_backend/app/config/database.py`
**Unique Mission:** DB indexing strategy, caching layers, query performance optimization, database migrations.
**Your Tasks:**
- DB indexing strategy for MongoDB collections
- Caching layers (Redis, memory cache)
- Query performance optimization
- Database migrations for both local & cloud environments
- Backup + snapshot flow for persistence across devices
- Unique focus: ensures the desktop IDE can run "offline-first" without breaking sync when internet returns
- Redis caching for leaderboards and real-time data
- InfluxDB integration for time-series analytics
**Files to Work On:**
- `deepiri/services/user-service/src/timeSeriesService.js` - Time-series tracking
- `deepiri/services/analytics-service/src/timeSeriesAnalytics.js` - InfluxDB analytics
- `deepiri/python_backend/app/database/models.py` - Database models
- `deepiri/python_backend/app/utils/cache.py` - Caching utilities
**Files to Create:**
- `deepiri/python_backend/app/database/optimization.py` (if needed)
- `deepiri/python_backend/app/database/migrations/` (create directory)
- `deepiri/services/*/src/database/backup.js` (create in each service if needed)
- `deepiri/services/*/src/database/offline_sync.js` - Offline-first sync logic

---

### FullStack Engineer 1 (AI)
**Reports to:** Backend Lead
**Location:** `deepiri/frontend/src/pages/` & `deepiri/python_backend/app/routes/`
**Stack:** React, TypeScript, FastAPI, REST APIs, Real-time Updates
**Start Here:**
- Productivity chat: `deepiri/frontend/src/pages/ProductivityChat.jsx`
- Challenge routes: `deepiri/python_backend/app/routes/challenge.py`
**Unique Mission:** Builds UI hooks for AI interactions (cloud), visualizes model outputs + reasoning flow.
**Your Tasks:**
- Builds UI hooks for AI interactions (cloud)
- Visualizes model outputs + reasoning flow
- Builds the real-time AI feedback loop for microservices IDE
- UI → AI service API connectivity
- Challenge flow integration between cloud UI and backend
- Real-time AI response visualization
- AI agent status and progress indicators
**Files to Create:**
- `deepiri/frontend/src/components/ChallengeGenerator.jsx`
- `deepiri/frontend/src/components/AIResponseViewer.jsx`
- `deepiri/frontend/src/components/ModelOutputVisualization.jsx`
- `deepiri/frontend/src/services/challengeApi.js`

---

### FullStack Engineer 2 - Gamification UI
**Reports to:** Backend Lead
**Location:** `deepiri/frontend/src/components/gamification/` & `deepiri/services/gamification-service/`
**Stack:** React, TypeScript, Socket.IO, Real-time Updates, Animations
**Start Here:**
- Gamification service: `deepiri/services/gamification-service/server.js` (Port 5003)
- Route handlers: `deepiri/services/gamification-service/src/index.js`
- Services: `deepiri/services/gamification-service/src/multiCurrencyService.js`, `eloLeaderboardService.js`, `badgeSystemService.js`
- Frontend components: `deepiri/frontend/src/components/`
**Unique Mission:** Designs gamified progress flows, achievement animations, leaderboard systems.
**Your Tasks:**
- Designs gamified progress flows
- Achievement animations and visual effects
- Leaderboard systems UI
- Social or collaborative challenge UI
- UI effects optimized for both web and native UI frameworks
- Badge and reward visualization
- Streak tracking UI components
- Progress bar and XP visualization
**Files to Work On:**
- `deepiri/services/gamification-service/server.js` - Service server (port 5003)
- `deepiri/services/gamification-service/src/index.js` - Route handlers
- `deepiri/services/gamification-service/src/multiCurrencyService.js` - Multi-currency
- `deepiri/services/gamification-service/src/eloLeaderboardService.js` - ELO leaderboard
- `deepiri/services/gamification-service/src/badgeSystemService.js` - Badge system
**Files to Create:**
- `deepiri/frontend/src/components/gamification/ProgressTracker.jsx`
- `deepiri/frontend/src/components/gamification/BadgeAnimation.jsx`
- `deepiri/frontend/src/components/gamification/Leaderboard.jsx`
- `deepiri/frontend/src/components/gamification/SocialFeatures.jsx`

---

### FullStack Engineer 3 - Integrations UI
**Reports to:** Backend Lead
**Location:** `deepiri/frontend/src/pages/integrations/` & `deepiri/services/integration-service/`
**Stack:** React, TypeScript, Node.js, OAuth2, REST APIs
**Start Here:**
- Integration service: `deepiri/services/integration-service/server.js` (Port 5006)
- Route handlers: `deepiri/services/integration-service/src/index.js`
- Webhook service: `deepiri/services/integration-service/src/webhookService.js`
- Frontend: `deepiri/frontend/src/pages/`
**Unique Mission:** Builds external integrations UI, OAuth connection dashboard, real-time sync status visualization.
**Your Tasks:**
- Builds external integrations UI
- OAuth connection dashboard
- Real-time sync status visualization
- Error reporting dashboard
- Device connection settings for native desktop IDE
- Integration status and health monitoring UI
- Webhook configuration interface
**Files to Work On:**
- `deepiri/services/integration-service/server.js` - Service server
- `deepiri/services/integration-service/src/index.js` - Route handlers
- `deepiri/services/integration-service/src/webhookService.js` - Webhook processing
**Files to Create:**
- `deepiri/frontend/src/pages/integrations/Dashboard.jsx`
- `deepiri/frontend/src/components/integrations/OAuthFlow.jsx`
- `deepiri/frontend/src/components/integrations/SyncMonitor.jsx`
- `deepiri/frontend/src/components/integrations/ConfigInterface.jsx`

---

### FullStack Engineer 4 - Analytics UI
**Reports to:** Backend Lead
**Location:** `deepiri/frontend/src/pages/analytics/` & `deepiri/services/analytics-service/`
**Stack:** React, TypeScript, Chart.js/D3.js, Node.js, InfluxDB, Real-time Data
**Start Here:**
- Analytics service: `deepiri/services/analytics-service/server.js` (Port 5004)
- Route handlers: `deepiri/services/analytics-service/src/index.js`
- Services: `deepiri/services/analytics-service/src/timeSeriesAnalytics.js`, `behavioralClustering.js`, `predictiveModeling.js`
- Frontend: `deepiri/frontend/src/pages/`
**Unique Mission:** Data visualizations, productivity metrics UI, real-time analytics integration.
**Your Tasks:**
- Data visualizations for productivity metrics
- Productivity metrics UI
- Real-time analytics integration with backend's data pipeline
- Export tools for analytics data
- Insight recommendation layout
- Time-series charts and graphs
- User behavior visualization
**Files to Work On:**
- `deepiri/services/analytics-service/server.js` - Service server (port 5004)
- `deepiri/services/analytics-service/src/index.js` - Route handlers
- `deepiri/services/analytics-service/src/timeSeriesAnalytics.js` - InfluxDB analytics
- `deepiri/services/analytics-service/src/behavioralClustering.js` - Clustering
- `deepiri/services/analytics-service/src/predictiveModeling.js` - Predictive models
**Files to Create:**
- `deepiri/frontend/src/pages/analytics/Dashboard.jsx`
- `deepiri/frontend/src/components/analytics/ProductivityChart.jsx`
- `deepiri/frontend/src/components/analytics/InsightRecommendations.jsx`

---

### Systems Architect 1 - Microservices Communication
**Reports to:** Backend Lead
**Location:** `deepiri/services/` & `deepiri/docs/MICROSERVICES_SETUP.md`
**Stack:** System Design, Microservices Patterns, API Gateway, Service Mesh
**Start Here:**
- Architecture doc: `deepiri/docs/MICROSERVICES_SETUP.md` - **READ THIS FIRST!**
- API Gateway: `deepiri/services/api-gateway/server.js` (Port 5000)
- Docker Compose: `deepiri/docker-compose.dev.yml`
**Unique Mission:** Establishes new microservices communication patterns, Improves microservice communication, cloud scaling approach, native-to-cloud handshake protocol.
**Your Tasks:**
- Establishes microservices communication patterns
- Cloud scaling approach for 9+ services
- Native-to-cloud handshake protocol
- Versioning strategy for backend services
- API Gateway routing and load balancing
- Service discovery and health checks
- Inter-service communication protocols
**Files to Work On:**
- `deepiri/services/api-gateway/server.js` - API Gateway routing
- `deepiri/services/api-gateway/src/index.js` - Route definitions
- `deepiri/docker-compose.dev.yml` - Service orchestration
**Files to Create:**
- `deepiri/architecture/microservices_design.md` (if needed)
- `deepiri/architecture/service_communication.md` (if needed)
- `deepiri/services/api-gateway/src/service_discovery.js` (if needed)
- `deepiri/services/api-gateway/src/load_balancer.js` (if needed)
- `deepiri/architecture/native_cloud_protocol.md` - Native-to-cloud handshake

---

### Systems Architect 2 - Event Bus & Streaming
**Reports to:** Backend Lead
**Location:** `deepiri/services/` (Event-Driven Architecture)
**Stack:** Kafka, RabbitMQ, Event Sourcing, CQRS, Message Queues
**Start Here:**
- Services: `deepiri/services/`
**Unique Mission:** Event bus architecture, real-time stream processing, message queue logic.
**Your Tasks:**
- Event bus architecture (Redis Streams/Kafka)
- Real-time stream processing
- Message queue logic for async communication
- Distributed data consistency
- Pipeline design for analytics → ML engineers
- Event-driven architecture patterns
- Stream processing for user events
**Files to Create:**
- `deepiri/architecture/event_driven_design.md`
- `deepiri/services/event-bus/` (create service)
- `deepiri/services/event-bus/src/kafka_producer.js`
- `deepiri/services/event-bus/src/event_processor.js`

---

### Systems Architect 3 - Security & Compliance
**Reports to:** Backend Lead
**Location:** `deepiri/services/` & `deepiri/python_backend/app/middleware/`
**Stack:** Security Architecture, OAuth2, JWT, Encryption, API Security
**Start Here:**
- Middleware: `deepiri/python_backend/app/middleware/`
- Services: `deepiri/services/`
**Unique Mission:** Authentication flows, token security, data privacy requirements, API hardening.
**Your Tasks:**
- Authentication flows (JWT, OAuth)
- Token security and refresh mechanisms
- Data privacy requirements (GDPR compliance)
- API hardening and security best practices
- Compliance alignment (HIPAA/GDPR if needed)
- Rate limiting and DDoS protection
- Security audit logging
**Files to Create:**
- `deepiri/architecture/security_design.md`
- `deepiri/services/auth-service/src/encryption.js`
- `deepiri/services/auth-service/src/api_security.js`
- `deepiri/python_backend/app/middleware/security.py`

---

### Systems Architect 4 - Scaling & High Availability
**Reports to:** Backend Lead
**Location:** `deepiri/services/websocket-service/` & `deepiri/services/gamification-service/`
**Stack:** Real-time Systems, WebSockets, Scalability, Multiplayer Architecture
**Start Here:**
- WebSocket service: `deepiri/services/websocket-service/`
- Gamification service: `deepiri/services/gamification-service/`
**Unique Mission:** Scaling live sessions, game state management, global replication strategies.
**Your Tasks:**
- Scaling live sessions (WebSocket connections)
- Game state management for challenges
- Global replication strategies
- Load balancing across services
- Failover systems for cloud IDE during peak usage
- Database replication and sharding
- CDN integration for static assets
**Files to Create:**
- `deepiri/architecture/multiplayer_scaling.md`
- `deepiri/services/websocket-service/src/session_scaling.js`
- `deepiri/services/websocket-service/src/game_state_manager.js`
- `deepiri/services/websocket-service/src/disaster_recovery.js`

---

### Systems Architect Intern (Officially: Systems Architect)
**Reports to:** Backend Lead
**Location:** `deepiri/architecture/` & `deepiri/services/`
**Stack:** System Design, Documentation, Architecture Patterns
**Start Here:**
- Architecture docs: `deepiri/docs/MICROSERVICES_SETUP.md`
- Services: `deepiri/services/`
**Unique Mission:** Supporting system architecture design, documentation, prototyping.
**Your Tasks:**
- Documents architecture decisions
- Prototypes small architectural experiments
- Validates communication flows between services
- Creates architecture diagrams
- Assists with system design documentation
**Files to Create:**
- `deepiri/architecture/patterns.md`
- `deepiri/architecture/service_templates.md`
- `deepiri/architecture/design_reviews.md`

---

### Systems Engineer 1 - End-to-End Integration
**Reports to:** Product Lead / Founder
**Location:** `deepiri/` (Cross-system integration)
**Stack:** System Integration, End-to-End Testing, API Integration, System Validation
**Start Here:**
- Root: `deepiri/`
- Integration tests: `deepiri/python_backend/tests/integration/`
**Unique Mission:** Ensure the entire Deepiri system (web IDE, native desktop IDE, AI services, backend, and cloud) works seamlessly as an integrated product.
**Your Tasks:**
- Conduct end-to-end flow validation across web and desktop platforms
- Perform system-wide behavior testing (AI agents, task flow, challenge gamification)
- Coordinate with backend and AI Systems teams to detect integration issues
- Maintain test matrices for multi-service and multi-platform scenarios
- Report and prioritize system-level anomalies
- Integration testing across all microservices
- Validate AI service integration with backend
**Files to Create:**
- `deepiri/tests/integration/full_system_test.py`
- `deepiri/tests/integration/ai_backend_integration.py`
- `deepiri/scripts/system_health_check.sh`

---

### Systems Engineer 2 - Visual Documentation
**Reports to:** Product Lead / Founder
**Location:** `deepiri/docs/` & `deepiri/` (Visual Documentation)
**Stack:** Documentation, Diagramming, Visual Design, Planning Tools
**Start Here:**
- Documentation: `deepiri/docs/`
- Architecture: `deepiri/docs/SYSTEM_ARCHITECTURE.md`
**Unique Mission:** Support the Product Lead in visualizing, planning, and documenting system flows without touching software.
**Your Tasks:**
- Create physical diagrams for workflows, data flows, and user journeys
- Produce schematics for device-agent interactions (native IDE)
- Visualize task and challenge logic for planning sessions
- Maintain visual documentation for onboarding and team alignment
- Assist Product Lead with whiteboarding, brainstorming product flow diagrams, and planning sessions
**Files to Create:**
- `deepiri/docs/diagrams/workflows/` - Workflow diagrams
- `deepiri/docs/diagrams/data_flows/` - Data flow diagrams
- `deepiri/docs/diagrams/user_journeys/` - User journey maps
- `deepiri/docs/diagrams/device_agent/` - Device-agent interaction schematics

---

### Platform Engineer 1 (Lead)
**Reports to:** Backend Lead
**Location:** `deepiri/platform/` & `deepiri/.github/`
**Stack:** Internal Developer Platform, CI/CD, Developer Tooling, Productivity Tools
**Start Here:**
- Platform directory: `deepiri/platform/` (create if needed)
- CI/CD: `deepiri/.github/workflows/`
- Scripts: `deepiri/scripts/*.sh`
- Environment setup: `deepiri/ENVIRONMENT_SETUP.md`
- Dev start: `deepiri/dev-start.js`
**Unique Mission:** Developer tooling, internal CLIs and workflows, productivity automation.
**Your Tasks:**
- Developer tooling for local development
- Internal CLIs and workflows
- Productivity automation for development team
- Inter-service debugging tools
- Local development environment setup
- Developer onboarding tooling
**Files to Create:**
- `deepiri/platform/developer_portal/`
- `deepiri/.github/workflows/ci.yml`
- `deepiri/platform/tooling/`
- `deepiri/platform/cli/` - Internal CLI tools

---

### Platform Engineer 2 - Infrastructure as Code
**Reports to:** Backend Lead / Platform Engineer 1
**Location:** `deepiri/infrastructure/` & `deepiri/docker-compose.yml`
**Stack:** Terraform, Infrastructure as Code, Cloud Provisioning, Docker
**Start Here:**
- Docker compose: `deepiri/docker-compose.yml`
- Infrastructure: `deepiri/infrastructure/` (create)
- Kubernetes: `deepiri/ops/k8s/*.yaml`
**Unique Mission:** Infrastructure as Code templates, automated provisioning, security practices.
**Your Tasks:**
- Infrastructure as Code templates (Terraform/CloudFormation)
- Automated provisioning of dev and prod environments
- Ensures both web + native IDE share configuration pipelines
- Embed security practices into the platform
- Manage access controls
- Ensure the platform adheres to compliance standards
- Kubernetes configuration and deployment
**Files to Create:**
- `deepiri/infrastructure/terraform/`
- `deepiri/infrastructure/kubernetes/`
- `deepiri/infrastructure/docker/`

---

### Cloud / Infrastructure Engineer 1 (Lead)
**Reports to:** Backend Lead
**Location:** `deepiri/infrastructure/` & `deepiri/docker-compose.yml`
**Stack:** AWS/GCP/Azure, Kubernetes, Cloud Resource Management, Networking
**Start Here:**
- Docker compose: `deepiri/docker-compose.yml`
- Infrastructure: `deepiri/infrastructure/`
**Unique Mission:** Cloud resource management, networking, cost optimization.
**Your Tasks:**
- Cloud compute cluster management
- Networking layout for microservices
- Cost optimization for cloud resources
- Resource allocation and scaling
- Cloud provider integration (AWS/GCP/Azure)
- Network security and VPC configuration
**Files to Create:**
- `deepiri/infrastructure/cloud/aws/`
- `deepiri/infrastructure/cloud/gcp/`
- `deepiri/infrastructure/networking/`
- `deepiri/infrastructure/cost_optimization.md`

---

### Cloud / Infrastructure Engineer 2 - Observability
**Reports to:** Backend Lead
**Location:** `deepiri/infrastructure/monitoring/` & `deepiri/infrastructure/security/`
**Stack:** Prometheus, Grafana, Cloud Security, Monitoring Tools
**Start Here:**
- Infrastructure: `deepiri/infrastructure/`
- Prometheus: `deepiri/ops/prometheus/prometheus.yml`
- Fluentd: `deepiri/ops/fluentd/*.conf`
**Unique Mission:** Security logging, resource utilization, bottleneck identification, observability dashboards.
**Your Tasks:**
- Security logging and audit trails
- Resource utilization monitoring
- Bottleneck identification and optimization
- Observability dashboards (Grafana, Prometheus)
- Performance monitoring and alerting
- Log aggregation and analysis
**Files to Create:**
- `deepiri/infrastructure/monitoring/prometheus.yml`
- `deepiri/infrastructure/monitoring/grafana/`
- `deepiri/infrastructure/security/security_scanning.sh`
- `deepiri/infrastructure/monitoring/alerts.yml`

---

### Cloud / Infrastructure Engineer 3 - High Availability
**Reports to:** Backend Lead
**Location:** `deepiri/infrastructure/disaster_recovery/` & `deepiri/infrastructure/backup/`
**Stack:** Backup Systems, Disaster Recovery, High Availability, Failover
**Start Here:**
- Infrastructure: `deepiri/infrastructure/`
**Unique Mission:** Backup strategy, multi-region availability, failover systems.
**Your Tasks:**
- Backup strategy for databases and user data
- Multi-region availability
- Failover systems for high availability
- Native IDE → cloud sync recovery mechanisms
- Disaster recovery planning
- Data replication across regions
**Files to Create:**
- `deepiri/infrastructure/disaster_recovery/plan.md`
- `deepiri/infrastructure/backup/backup_strategy.md`
- `deepiri/infrastructure/ha/high_availability.yml`
- `deepiri/infrastructure/failover/failover_config.yml`
- `deepiri/infrastructure/sync_recovery/` - Native IDE sync recovery

---

### DevOps Engineer
**Reports to:** Backend Lead
**Location:** `deepiri/.github/workflows/` & `deepiri/infrastructure/`
**Stack:** CI/CD, Docker, Kubernetes, Monitoring, Observability
**Start Here:**
- CI/CD: `deepiri/.github/workflows/`
- Docker: `deepiri/docker-compose.yml`
**Unique Mission:** CI/CD, monitoring, observability, deployment pipelines.
**Your Tasks:**
- CI/CD pipeline configuration
- Monitoring and alerting setup
- Observability infrastructure
- Deployment pipelines for both web services and native update service
- Automated testing in CI/CD
- Release management and versioning
**Files to Create:**
- `deepiri/.github/workflows/deploy.yml`
- `deepiri/infrastructure/ci_cd/`
- `deepiri/infrastructure/monitoring/observability.yml`
- `deepiri/infrastructure/native_updates/` - Native update service

---

### Backend Intern 1 (MLOps Intern)
**Reports to:** Backend Lead
**Location:** `deepiri/services/*/tests/` & `deepiri/.github/workflows/`
**Stack:** Testing, CI/CD, Test Automation
**Start Here:**
- Services: `deepiri/services/`
- CI/CD: `deepiri/.github/workflows/`
**Unique Mission:** Microservice testing, CI/CD pipeline support, API validation.
**Your Tasks:**
- Microservice testing
- Supports CI/CD pipeline
- Automates test workflows and API validation
- Integration testing for services
- Test coverage improvement
**Files to Create:**
- `deepiri/services/*/tests/` (in each service)
- `deepiri/.github/workflows/test.yml`

---

### Backend Intern 2 - Documentation
**Reports to:** Backend Lead
**Location:** `deepiri/services/*/docs/` & `deepiri/services/*/src/`
**Stack:** Documentation, API Documentation, Logging
**Start Here:**
- Services: `deepiri/services/`
**Unique Mission:** Writes request/response schemas, improves logging clarity, builds internal documentation.
**Your Tasks:**
- Writes request/response schemas
- Improves logging clarity
- Builds internal documentation and debug logs panel
- API documentation maintenance
- Code documentation and comments
**Files to Create:**
- `deepiri/services/*/docs/API.md` (for each service)
- `deepiri/services/*/src/logging.js`
- `deepiri/docs/api_overview.md`
- `deepiri/services/*/src/schemas/` - Request/response schemas

---

### Backend Intern 3 - Performance Testing
**Reports to:** Backend Lead
**Location:** `deepiri/services/*/tests/performance/` & `deepiri/services/*/src/`
**Stack:** Performance Testing, Load Testing, Bug Fixes
**Start Here:**
- Services: `deepiri/services/`
**Unique Mission:** Stress tests APIs, identifies slow endpoints, backend bug fixing.
**Your Tasks:**
- Stress tests APIs
- Identifies slow endpoints
- Backend bug fixing
- Performance reports
- Load testing and optimization
**Files to Create:**
- `deepiri/services/*/tests/performance/load_test.js`
- `deepiri/services/*/tests/performance/stress_test.js`
- `deepiri/scripts/performance_test_suite.sh`

---

## FRONTEND & UI TEAM

### Frontend Lead
**Reports to:** Product Lead
**Location:** `deepiri/frontend/`
**Stack:** React, TypeScript, Vite, Tailwind CSS, UX/UI Design
**Start Here:**
- Frontend: `deepiri/frontend/`
- App: `deepiri/frontend/src/App.jsx`
**Unique Mission:** Supervises web/frontend architecture, design consistency, and integration of UI with backend and cloud AI services. Reviews and merges code for frontend team.
**Your Tasks:**
- Supervises web/frontend architecture (React 18 + Vite)
- Design consistency across web and native desktop app
- Integration of UI with backend and cloud AI services
- UI theme direction for both web + native desktop app
- Review and merge code from Frontend Engineers
- Establish frontend coding standards
- Coordinate with Backend Lead on API integration
- Ensure responsive design and accessibility
**Files to Review:**
- `deepiri/frontend/src/App.jsx`
- `deepiri/frontend/src/components/`
- `deepiri/frontend/src/pages/`

---

### Graphic Designer
**Reports to:** Product Lead
**Location:** `deepiri/frontend/src/assets/` & `deepiri/frontend/public/`
**Stack:** Design Tools (Figma, Adobe), SVG, Branding
**Start Here:**
- Assets: `deepiri/frontend/src/assets/`
- Public: `deepiri/frontend/public/`
**Unique Mission:** Logo Design, Branding, Visual Identity.
**Your Tasks:**
- Logo design and brand identity
- Visual branding guidelines
- UI/UX design mockups
- Design system and style guide
- Icon and asset creation
- Brand consistency across platforms
**Files to Create:**
- `deepiri/frontend/src/assets/logo.svg`
- `deepiri/frontend/src/assets/branding/`
- `deepiri/frontend/public/favicon.ico`
- `deepiri/frontend/src/styles/brand.css`

---

### Frontend Engineer 1 - Core UI Foundation
**Reports to:** Frontend Lead
**Location:** `deepiri/frontend/src/pages/` & `deepiri/frontend/src/components/`
**Stack:** React, TypeScript, Firebase, Forms, Dashboards
**Start Here:**
- Pages: `deepiri/frontend/src/pages/`
- Components: `deepiri/frontend/src/components/`
**Unique Mission:** Builds pages, dashboards, forms, authentication flows for web app.
**Your Tasks:**
- Builds pages, dashboards, forms
- Authentication flows for web app
- UI foundations that the other engineers build on
- Core React components and layouts
- Routing and navigation
- Form validation and error handling
**Files to Create:**
- `deepiri/frontend/src/pages/Dashboard.jsx`
- `deepiri/frontend/src/components/forms/`
- `deepiri/frontend/src/services/firebase.js`
- `deepiri/frontend/src/components/forms/FormValidation.jsx`

---

### Frontend Engineer 2 - AI Visualization
**Reports to:** Frontend Lead
**Location:** `deepiri/frontend/src/components/charts/` & `deepiri/frontend/src/pages/analytics/`
**Stack:** React, TypeScript, Chart.js, D3.js, Data Visualization
**Start Here:**
- Analytics: `deepiri/frontend/src/pages/analytics/`
- Components: `deepiri/frontend/src/components/`
**Unique Mission:** AI reasoning visualization, advanced charting, multimodal rendering.
**Your Tasks:**
- AI reasoning visualization
- Advanced charting for analytics
- Multimodal rendering (text, images, code)
- Interactive charts for analytics tools
- Real-time AI response visualization
- Model output display components
**Files to Create:**
- `deepiri/frontend/src/components/charts/ProductivityChart.jsx`
- `deepiri/frontend/src/components/charts/AIInsightsChart.jsx`
- `deepiri/frontend/src/pages/analytics/Dashboard.jsx`
- `deepiri/frontend/src/components/charts/ChartLibrary.jsx`

---

### Frontend Engineer 3 - Gamification UI
**Reports to:** Frontend Lead
**Location:** `deepiri/frontend/src/components/gamification/`
**Stack:** React, TypeScript, CSS Animations, Gamification UI
**Start Here:**
- Components: `deepiri/frontend/src/components/`
**Unique Mission:** Badges, animated avatars, progress bars, challenge animations.
**Your Tasks:**
- Badges and achievement UI
- Animated avatars
- Progress bars and XP visualization
- Challenge animations
- Cross-platform gamification components for web + desktop UI frameworks
- Streak tracking UI
- Reward animations
**Files to Create:**
- `deepiri/frontend/src/components/gamification/Badge.jsx`
- `deepiri/frontend/src/components/gamification/ProgressBar.jsx`
- `deepiri/frontend/src/components/gamification/Avatar.jsx`
- `deepiri/frontend/src/components/gamification/Animations.jsx`

---

### Frontend Engineer 4 - Performance & PWA
**Reports to:** Frontend Lead
**Location:** `deepiri/frontend/` (Performance & PWA)
**Stack:** React, TypeScript, PWA, Performance Optimization, Service Workers
**Start Here:**
- Frontend: `deepiri/frontend/`
- Config: `deepiri/frontend/vite.config.js`
**Unique Mission:** SPA optimization, PWA support, preloading strategies, offline-first UI cache.
**Your Tasks:**
- SPA optimization (React performance)
- PWA support for mobile
- Preloading strategies
- Bridges web UI patterns with native desktop application frontend
- Offline-first UI cache
- Code splitting and lazy loading
- Bundle size optimization
**Files to Create:**
- `deepiri/frontend/public/service-worker.js`
- `deepiri/frontend/src/utils/performance.js`
- `deepiri/frontend/vite.config.js` (optimize)
- `deepiri/frontend/public/manifest.json`

---

### Frontend Intern 1
**Reports to:** Frontend Lead
**Location:** `deepiri/frontend/src/components/` & `deepiri/frontend/tests/`
**Stack:** React, Testing Library, Jest, Component Development
**Start Here:**
- Components: `deepiri/frontend/src/components/`
- Tests: `deepiri/frontend/tests/`
**Unique Mission:** Component polish, UI tests, bug fixes, learns the architecture.
**Your Tasks:**
- Component polish and refinement
- UI tests (Jest, React Testing Library)
- Bug fixes and small improvements
- Learns the architecture and contributes to smaller UI pieces
- Accessibility improvements
- Cross-browser testing
**Files to Create:**
- `deepiri/frontend/tests/components/`
- `deepiri/frontend/src/components/common/Button.jsx`
- `deepiri/frontend/src/components/common/Input.jsx`

---

## SECURITY, SUPPORT & QA TEAM

### IT Lead
**Reports to:** Founder
**Location:** `deepiri/infrastructure/security/` & `deepiri/services/auth-service/`
**Stack:** Security Architecture, Network Defense, Cloud Security, Secure Microservices
**Start Here:**
- Infrastructure: `deepiri/infrastructure/`
- Auth service: `deepiri/services/auth-service/`
**Unique Mission:** Infrastructure support, organizational tech support, Network defense, secure microservices, cloud security.
**Your Tasks:**
- Infrastructure support for development and production
- Organizational tech support for employees
- Network defense and security monitoring
- Secure microservices architecture
- Cloud security and compliance
- Security incident response
- Access control and identity management
- Security policy enforcement
**Files to Create:**
- `deepiri/infrastructure/security/network_defense.md`
- `deepiri/infrastructure/security/cloud_security.yml`
- `deepiri/services/auth-service/src/security.js`

---

### IT Internal Support
**Reports to:** IT Lead
**Location:** `deepiri/docs/internal/` & `deepiri/scripts/onboarding/`
**Stack:** Internal Tools, Documentation, Onboarding Systems
**Start Here:**
- Docs: `deepiri/docs/`
**Unique Mission:** Employee tech support, onboarding/offboarding, software/hardware provisioning.
**Your Tasks:**
- Employee tech support
- Employee onboarding/offboarding
- Software/hardware provisioning
- Organizational tech support
- Development environment setup
- IT asset management
**Files to Create:**
- `deepiri/docs/internal/onboarding.md`
- `deepiri/docs/internal/tech_support.md`
- `deepiri/scripts/onboarding/setup.sh`

---

### IT External Support
**Reports to:** IT Lead
**Location:** `deepiri/docs/user/` & `deepiri/frontend/src/pages/support/`
**Stack:** User Support, Documentation, Help Systems
**Start Here:**
- Docs: `deepiri/docs/`
- Frontend: `deepiri/frontend/src/pages/`
**Unique Mission:** User tech support for Deepiri platform users.
**Your Tasks:**
- User tech support
- Customer service for technical issues
- User onboarding assistance
- Bug report triage and escalation
- User documentation and help guides
**Files to Create:**
- `deepiri/docs/user/getting_started.md`
- `deepiri/frontend/src/pages/support/HelpCenter.jsx`
- `deepiri/frontend/src/pages/support/Contact.jsx`

---

### Security Operations/ Support Engineer 1
**Reports to:** IT Lead / Founder
**Location:** `deepiri/infrastructure/security/` & `deepiri/scripts/security/`
**Stack:** Security Monitoring, Resource Management, Vulnerability Assessment
**Start Here:**
- Infrastructure: `deepiri/infrastructure/security/`
**Unique Mission:** Resource monitoring (Github, Discord), security operations.
**Your Tasks:**
- Resource monitoring (Github, Discord)
- Security monitoring and alerting
- Vulnerability scanning
- Security incident investigation
- Access control monitoring
**Files to Create:**
- `deepiri/docs/support/monitoring.md`
- `deepiri/scripts/monitoring/resource_check.sh`
- `deepiri/docs/support/alerts.md`

---

### Security Operations/ Support Engineer 2
**Reports to:** IT Lead
**Location:** `deepiri/infrastructure/security/` & `deepiri/scripts/security/`
**Stack:** Security Monitoring, Resource Management, Vulnerability Assessment
**Start Here:**
- Infrastructure: `deepiri/infrastructure/security/`
**Unique Mission:** Resource monitoring (Github, Discord), security operations.
**Your Tasks:**
- Resource monitoring (Github, Discord)
- Security monitoring and alerting
- Vulnerability scanning
- Security incident investigation
- Access control monitoring
**Files to Create:**
- `deepiri/scripts/security/dependency_scan.sh`
- `deepiri/scripts/security/vulnerability_check.sh`
- `deepiri/infrastructure/security/security_policy.md`
- `.github/dependabot.yml` (update)

---

### Security Operations Analyst
**Reports to:** IT Lead
**Location:** `deepiri/infrastructure/security/` & `deepiri/scripts/security/`
**Stack:** Security Scanning, Dependency Management, Vulnerability Assessment
**Start Here:**
- Infrastructure: `deepiri/infrastructure/security/`
**Unique Mission:** Resource monitoring (Github, Discord), checking cloud security, platform security, checking Dependabot alerts.
**Your Tasks:**
- Resource monitoring (Github, Discord)
- Checking cloud security configurations
- Platform security assessment
- Checking Dependabot alerts for dependency files
- Security vulnerability analysis
- Security compliance checking
- Dependency security scanning

---

### QA Lead
**Reports to:** Product Lead
**Location:** `deepiri/tests/` & `deepiri/qa/`
**Stack:** Test Planning, Integration Testing, Test Strategy
**Start Here:**
- Tests: `deepiri/tests/`
- QA: `deepiri/qa/` (create)
**Unique Mission:** Ensure product quality across both web and native IDE, guiding QA engineers in testing strategy and coverage. Reviews and merges code for QA team.
**Your Tasks:**
- Create test plans for new features and releases
- Coordinate integration testing between frontend, backend, and AI systems
- Oversee regression testing to maintain system stability
- Define QA standards and testing protocols
- Act as primary QA liaison with Product Lead
- Review and merge code for QA team and other teams
- Establish testing automation strategy
- Coordinate with all teams on quality standards
**Files to Create:**
- `deepiri/qa/test_plans.md`
- `deepiri/qa/integration_test_plan.md`
- `deepiri/qa/regression_test_plan.md`

---

### QA Engineer 1 - Manual Testing
**Reports to:** QA Lead
**Location:** `deepiri/tests/manual/` & `deepiri/qa/manual/`
**Stack:** Manual Testing, User Acceptance Testing, Test Cases
**Start Here:**
- Tests: `deepiri/tests/`
**Unique Mission:** Test features from a user perspective to ensure functionality and usability.
**Your Tasks:**
- Execute manual test cases across web and native IDE
- Perform user acceptance testing (UAT) on new releases
- Verify bug fixes and report anomalies to QA Lead
- Provide feedback on user experience and workflow issues
- Validate AI agent output alignment with intended task guidance
- Test challenge generation and gamification flows
- Validate external integrations (Notion, Trello, GitHub)
**Files to Create:**
- `deepiri/qa/manual/test_cases.md`
- `deepiri/qa/manual/uat_scenarios.md`
- `deepiri/qa/manual/bug_reports.md`

---

### QA Engineer 2 - Automation
**Reports to:** QA Lead
**Location:** `deepiri/tests/automation/` & `deepiri/qa/automation/`
**Stack:** Automation Testing, API Testing, Selenium, Playwright, Test Reporting
**Start Here:**
- Tests: `deepiri/tests/`
**Unique Mission:** Ensure system reliability and backend robustness through automation.
**Your Tasks:**
- Design and maintain automated test scripts for backend APIs
- Perform API endpoint validation for web and native IDE
- Track metrics on automation coverage and bug trends
- Report testing results to QA Lead and coordinate with backend team
- Automate repetitive regression and integration tests
- API testing with Postman/Newman or similar tools
- End-to-end testing with Playwright/Cypress
**Files to Create:**
- `deepiri/tests/automation/api_tests.js`
- `deepiri/tests/automation/e2e_tests.js`
- `deepiri/qa/automation/test_reports/`
- `deepiri/tests/automation/framework/`

---

### QA Intern 1
**Reports to:** QA Lead / Engineers
**Location:** `deepiri/tests/` & `deepiri/qa/`
**Stack:** Test Scripts, Bug Tracking, Basic QA Tasks
**Start Here:**
- Tests: `deepiri/tests/`
**Unique Mission:** Assist QA engineers with routine testing and documentation. KEEPS TRACK OF TESTING DOCUMENTATION.
**Your Tasks:**
- Write and maintain simple test scripts for manual testing
- Log, track, and document bugs in the tracking system
- Perform basic QA tasks such as cross-browser/device checks
- Support QA Engineer 1 in user acceptance testing
- Help maintain test case documentation
- **YOU KEEP TRACK OF THE TESTING DOCUMENTATION!**
**Files to Create:**
- `deepiri/tests/scripts/`
- `deepiri/qa/bug_tracking.md`
- `deepiri/tests/data/test_data.json`

---

### QA Intern 2
**Reports to:** QA Lead / Engineers
**Location:** `deepiri/tests/regression/` & `deepiri/qa/`
**Stack:** Regression Testing, Test Suite Maintenance, Environment Setup
**Start Here:**
- Tests: `deepiri/tests/`
**Unique Mission:** Support QA team in maintaining system stability and test environments.
**Your Tasks:**
- Maintain regression test suites for repeated releases
- Assist in setting up testing environments (web and native)
- Ensure test data consistency for repeated QA cycles
- Support QA Lead in executing integration regression runs
- Monitor environment health during automated test execution
- Test environment maintenance and cleanup
**Files to Create:**
- `deepiri/tests/regression/regression_suite.js`
- `deepiri/qa/environments/test_env_setup.md`
- `deepiri/tests/config/test_config.json`

---

## DESKTOP IDE TEAM

### Desktop IDE Developers
**Location:** `desktop-ide-deepiri/` (at project root, same level as deepiri/)
**Stack:** Tauri, Rust, React, TypeScript, SQLite, Local LLM
**Start Here:**
- Desktop IDE: `desktop-ide-deepiri/README.md`
- Tauri backend: `desktop-ide-deepiri/src-tauri/src/`
- React frontend: `desktop-ide-deepiri/src/renderer/`
**Your Tasks:**
- Desktop IDE development
- Tauri backend integration
- React frontend components
- Local LLM integration
- Desktop-specific features
**Key Files:**
- `desktop-ide-deepiri/src-tauri/src/main.rs`
- `desktop-ide-deepiri/src/renderer/App.jsx`
- `desktop-ide-deepiri/src/renderer/services/aiService.js`
- `desktop-ide-deepiri/src/renderer/components/`

---

## QUICK START CHECKLIST

1. **Find your role** in this document (Ctrl+F)
2. **Check your "Location"** - that's where you'll be working
3. **Review your "Stack"** - technologies you'll use
4. **Start with "Start Here"** files
5. **Work on "Your Tasks"** and "Files to Create/Work On"
6. **Ask questions** in team channels if stuck

---

## IMPORTANT LINKS

- **Main README:** `deepiri/README.md`
- **Microservices Architecture:** `deepiri/docs/MICROSERVICES_SETUP.md` - **READ THIS!**
- **Getting Started:** `deepiri/GETTING_STARTED.md`
- **Environment Variables:** `deepiri/ENVIRONMENT_VARIABLES.md`
- **Team Onboarding:** `deepiri/docs/*_TEAM_ONBOARDING.md`
- **AI Team Onboarding:** `deepiri/docs/AI_TEAM_ONBOARDING.md`
- **Backend Team Onboarding:** `deepiri/docs/BACKEND_TEAM_ONBOARDING.md`
- **Frontend Team Onboarding:** `deepiri/docs/FRONTEND_TEAM_ONBOARDING.md`
- **Desktop IDE README:** `desktop-ide-deepiri/README.md` (at project root)

## MICROSERVICES ARCHITECTURE NOTES

**All requests go through API Gateway (Port 5000):**
- Frontend connects to: `http://localhost:5000/api/*`
- API Gateway routes to appropriate microservice
- Each service runs on its own port (5001-5008)
- Python AI Service runs on port 8000

**Service Ports:**
- API Gateway: 5000
- User Service: 5001
- Task Service: 5002
- Gamification Service: 5003
- Analytics Service: 5004
- Notification Service: 5005
- Integration Service: 5006
- Challenge Service: 5007
- WebSocket Service: 5008
- Python AI Service: 8000

---

**Last Updated:** 2024
**Questions?** Contact your team lead or check team-specific READMEs.
