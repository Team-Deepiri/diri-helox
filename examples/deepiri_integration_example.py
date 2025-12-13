"""
Deepiri Platform Integration Examples
Shows how to collect data for all Deepiri features:
- Prompt-to-Tasks Engine (main differentiator)
- Tier 1: Intent Classification (50 predefined abilities)
- Tier 2: Role-based Ability Generation (dynamic abilities)
- Tier 3: RL Productivity Optimization
- Gamification System (Momentum, Streaks, Boosts, Objectives, Odysseys, Seasons)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from datetime import datetime
import time

from app.train.pipelines.data_collection_pipeline import get_data_collector
from app.services.command_router import get_command_router
from app.services.contextual_ability_engine import get_contextual_ability_engine

router = APIRouter()

# ============================================================================
# PROMPT-TO-TASKS ENGINE (Main Differentiator)
# ============================================================================

class PromptToTasksRequest(BaseModel):
    user_id: str
    user_role: str
    prompt: str
    project_type: Optional[str] = None

@router.post("/ai/prompt-to-tasks")
async def prompt_to_tasks_example(request: PromptToTasksRequest):
    """
    Main differentiator: Turn any idea into an execution plan.
    "Build login system" → tickets, subtasks, estimates
    """
    collector = get_data_collector()
    start_time = time.time()
    
    try:
        # Your prompt-to-tasks engine logic here
        # This is the core differentiator
        generated_tasks = [
            {
                "title": "Create login API endpoint",
                "subtasks": ["Design schema", "Implement auth", "Add tests"],
                "estimate": {"hours": 4, "complexity": "medium"}
            },
            {
                "title": "Build login UI component",
                "subtasks": ["Design mockup", "Implement form", "Add validation"],
                "estimate": {"hours": 3, "complexity": "medium"}
            }
        ]
        
        execution_plan = {
            "total_tasks": len(generated_tasks),
            "estimated_total_duration": sum(t.get('estimate', {}).get('hours', 0) for t in generated_tasks),
            "complexity": "medium"
        }
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Collect data
        collector.collect_prompt_to_tasks(
            user_id=request.user_id,
            user_role=request.user_role,
            prompt=request.prompt,
            generated_tasks=generated_tasks,
            execution_plan=execution_plan,
            project_type=request.project_type or "development",
            model_used="gpt-4",
            user_acceptance=None,  # User will accept/reject
            tasks_completed=None,  # Tracked later
            actual_duration=None,  # Tracked later
            user_feedback=None
        )
        
        # Also collect interaction
        collector.collect_interaction(
            user_id=request.user_id,
            user_role=request.user_role,
            action_type="prompt_to_tasks",
            context={
                "prompt": request.prompt,
                "tasks_generated": len(generated_tasks),
                "project_type": request.project_type
            },
            model_used="gpt-4",
            response_time_ms=latency_ms,
            success=True,
            feedback=None
        )
        
        return {
            "success": True,
            "tasks": generated_tasks,
            "execution_plan": execution_plan
        }
        
    except Exception as e:
        collector.collect_interaction(
            user_id=request.user_id,
            user_role=request.user_role,
            action_type="prompt_to_tasks",
            context={"prompt": request.prompt, "error": str(e)},
            model_used="gpt-4",
            response_time_ms=(time.time() - start_time) * 1000,
            success=False,
            feedback=None
        )
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TIER 1: Intent Classification (50 Predefined Abilities)
# ============================================================================

class CommandRequest(BaseModel):
    command: str
    user_id: str
    user_role: str = "software_engineer"
    context: Optional[Dict] = None
    min_confidence: float = 0.7

@router.post("/intelligence/route-command")
async def route_command_example(request: CommandRequest):
    """
    Tier 1: Classify user command to one of 50 predefined abilities.
    Uses fine-tuned BERT/DeBERTa for maximum reliability.
    """
    collector = get_data_collector()
    router = get_command_router()
    start_time = time.time()
    
    try:
        result = router.route(
            command=request.command,
            user_role=request.user_role,
            context=request.context or {},
            min_confidence=request.min_confidence
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Collect classification data
        collector.collect_classification(
            task_text=request.command,
            description=request.context.get('description') if request.context else None,
            prediction={
                'type': result.get('ability_id'),
                'complexity': result.get('complexity', 'medium'),
                'estimated_duration': result.get('duration', 30),
                'confidence': result.get('confidence', 0.0)
            },
            actual=None,  # User will provide feedback
            feedback=None,
            user_role=request.user_role
        )
        
        collector.collect_interaction(
            user_id=request.user_id,
            user_role=request.user_role,
            action_type="intent_classification",
            context={
                "command": request.command,
                "prediction": result,
                "confidence": result.get('confidence', 0)
            },
            model_used="deberta-v3-base",
            response_time_ms=latency_ms,
            success=result.get('confidence', 0) > request.min_confidence,
            feedback=None
        )
        
        return result
        
    except Exception as e:
        collector.collect_interaction(
            user_id=request.user_id,
            user_role=request.user_role,
            action_type="intent_classification",
            context={"command": request.command, "error": str(e)},
            model_used="deberta-v3-base",
            response_time_ms=(time.time() - start_time) * 1000,
            success=False,
            feedback=None
        )
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TIER 2: Role-based Ability Generation (Dynamic Abilities)
# ============================================================================

class AbilityGenerationRequest(BaseModel):
    user_id: str
    user_role: str
    user_command: str
    user_profile: Dict
    project_context: Optional[Dict] = None

@router.post("/intelligence/generate-ability")
async def generate_ability_example(request: AbilityGenerationRequest):
    """
    Tier 2: Generate dynamic, role-based abilities on-the-fly.
    Uses LLM + RAG for high creativity and flexibility.
    """
    collector = get_data_collector()
    engine = get_contextual_ability_engine()
    start_time = time.time()
    
    try:
        result = engine.generate_ability(
            user_id=request.user_id,
            user_command=request.user_command,
            user_profile=request.user_profile,
            project_context=request.project_context
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Collect ability generation data
        if result.get('success') and result.get('ability'):
            ability = result.get('ability', {})
            
            collector.collect_ability_generation(
                user_id=request.user_id,
                user_role=request.user_role,
                user_command=request.user_command,
                generated_ability=ability,
                rag_context=result.get('rag_context'),
                model_used=engine.llm_model_name,
                user_engagement=None,  # Tracked later
                ability_used=None,  # Tracked when user uses it
                completion_rate=None,
                performance_score=None
            )
        
        collector.collect_interaction(
            user_id=request.user_id,
            user_role=request.user_role,
            action_type="ability_generation",
            context={
                "command": request.user_command,
                "generated_ability": result.get('ability'),
                "user_profile": request.user_profile
            },
            model_used=engine.llm_model_name,
            response_time_ms=latency_ms,
            success=result.get('success', False),
            feedback=None
        )
        
        return result
        
    except Exception as e:
        collector.collect_interaction(
            user_id=request.user_id,
            user_role=request.user_role,
            action_type="ability_generation",
            context={"command": request.user_command, "error": str(e)},
            model_used=engine.llm_model_name,
            response_time_ms=(time.time() - start_time) * 1000,
            success=False,
            feedback=None
        )
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TIER 3: RL Productivity Recommendations
# ============================================================================

class ProductivityRecommendationRequest(BaseModel):
    user_id: str
    user_state: Dict

@router.post("/intelligence/productivity-recommendation")
async def productivity_recommendation_example(request: ProductivityRecommendationRequest):
    """
    Tier 3: RL-based productivity recommendations.
    Learns optimal actions to maximize long-term productivity.
    """
    collector = get_data_collector()
    # Your RL agent here
    start_time = time.time()
    
    try:
        # Simulate RL recommendation
        recommended_action = "activate_focus_boost"
        recommendation_type = "boost"
        expected_benefit = {
            "momentum_gain": 25,
            "time_saved_minutes": 30,
            "efficiency_increase": 0.15
        }
        
        latency_ms = (time.time() - start_time) * 1000
        
        collector.collect_productivity_recommendation(
            user_id=request.user_id,
            user_state=request.user_state,
            recommended_action=recommended_action,
            recommendation_type=recommendation_type,
            expected_benefit=expected_benefit,
            user_acceptance=None,  # User will accept/reject
            actual_benefit=None,  # Tracked after action
            reward_signal=None  # Calculated from outcome
        )
        
        return {
            "recommended_action": recommended_action,
            "type": recommendation_type,
            "expected_benefit": expected_benefit
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# GAMIFICATION SYSTEM DATA COLLECTION
# ============================================================================

@router.post("/gamification/objective-completed")
async def objective_completed_example(
    user_id: str,
    objective_id: str,
    title: str,
    momentum_reward: int,
    auto_detected: bool = False,
    odyssey_id: Optional[str] = None,
    season_id: Optional[str] = None
):
    """Collect Objective completion data."""
    collector = get_data_collector()
    
    collector.collect_objective_data(
        user_id=user_id,
        objective_id=objective_id,
        title=title,
        momentum_reward=momentum_reward,
        status="completed",
        completed_at=datetime.utcnow().isoformat(),
        auto_detected=auto_detected,
        odyssey_id=odyssey_id,
        season_id=season_id
    )
    
    # Also collect momentum event
    collector.collect_momentum_event(
        user_id=user_id,
        momentum_amount=momentum_reward,
        source=objective_id,
        source_type="objective",
        total_momentum=0,  # Get from user profile
        current_level=0,  # Get from user profile
        skill_category="tasks"
    )
    
    return {"success": True}


@router.post("/gamification/boost-activated")
async def boost_activated_example(
    user_id: str,
    boost_type: str,
    boost_source: str,
    credits_used: int,
    duration_minutes: int
):
    """Collect Boost activation data."""
    collector = get_data_collector()
    
    collector.collect_boost_usage(
        user_id=user_id,
        boost_type=boost_type,
        boost_source=boost_source,
        credits_used=credits_used,
        duration_minutes=duration_minutes,
        effectiveness_score=None,  # Calculated after boost expires
        tasks_completed=None,  # Tracked during boost
        time_saved_minutes=None  # Calculated after boost
    )
    
    return {"success": True}


@router.post("/gamification/streak-updated")
async def streak_updated_example(
    user_id: str,
    streak_type: str,
    streak_value: int,
    action: str,
    cashed_in: bool = False,
    boost_credits_earned: int = 0
):
    """Collect Streak event data."""
    collector = get_data_collector()
    
    collector.collect_streak_event(
        user_id=user_id,
        streak_type=streak_type,
        streak_value=streak_value,
        action=action,
        cashed_in=cashed_in,
        boost_credits_earned=boost_credits_earned
    )
    
    return {"success": True}


@router.post("/gamification/odyssey-created")
async def odyssey_created_example(
    user_id: str,
    odyssey_id: str,
    title: str,
    scale: str,
    organization_id: Optional[str] = None,
    season_id: Optional[str] = None
):
    """Collect Odyssey creation data."""
    collector = get_data_collector()
    
    collector.collect_odyssey_data(
        user_id=user_id,
        odyssey_id=odyssey_id,
        title=title,
        scale=scale,
        status="active",
        organization_id=organization_id,
        objectives_count=0,
        milestones_count=0,
        progress_percentage=0.0,
        season_id=season_id
    )
    
    return {"success": True}


@router.post("/gamification/season-created")
async def season_created_example(
    user_id: str,
    season_id: str,
    name: str,
    start_date: str,
    end_date: str,
    organization_id: Optional[str] = None
):
    """Collect Season creation data."""
    collector = get_data_collector()
    
    collector.collect_season_data(
        user_id=user_id,
        season_id=season_id,
        name=name,
        start_date=start_date,
        end_date=end_date,
        status="active",
        organization_id=organization_id,
        total_momentum_earned=0,
        objectives_completed=0,
        odysseys_completed=0
    )
    
    return {"success": True}


# ============================================================================
# RL TRAINING SEQUENCE COLLECTION
# ============================================================================

@router.post("/rl/collect-sequence")
async def collect_rl_sequence_example(
    user_id: str,
    state_data: Dict,
    action_taken: str,
    reward: float,
    next_state_data: Optional[Dict] = None,
    done: bool = False,
    episode_id: Optional[str] = None,
    step_number: int = 0
):
    """Collect RL training sequence for Tier 3."""
    collector = get_data_collector()
    
    collector.collect_rl_sequence(
        user_id=user_id,
        state_data=state_data,
        action_taken=action_taken,
        reward=reward,
        next_state_data=next_state_data,
        done=done,
        episode_id=episode_id,
        step_number=step_number
    )
    
    return {"success": True}


# ============================================================================
# USAGE NOTES
# ============================================================================

"""
DEEPIRI DATA COLLECTION INTEGRATION GUIDE:

1. PROMPT-TO-TASKS ENGINE (Main Differentiator)
   - Collect every prompt → tasks conversion
   - Track user acceptance and task completion
   - This is your core differentiator!

2. TIER 1: Intent Classification
   - Collect all command routing predictions
   - Get user feedback on predictions
   - Train fine-tuned BERT/DeBERTa model

3. TIER 2: Role-based Ability Generation
   - Collect all generated abilities
   - Track which abilities users actually use
   - Track engagement and completion rates
   - Train LLM + RAG system

4. TIER 3: RL Productivity Optimization
   - Collect state-action-reward sequences
   - Track recommendation acceptance
   - Track actual benefits vs expected
   - Train PPO agent

5. GAMIFICATION SYSTEM
   - Collect all momentum events
   - Track streak updates
   - Track boost usage and effectiveness
   - Track objective/odyssey/season data
   - Use for reward signal generation

QUICK INTEGRATION CHECKLIST:

□ Add prompt-to-tasks collection to your main engine
□ Add classification collection to command router
□ Add ability generation collection to ability engine
□ Add RL sequence collection to productivity agent
□ Add gamification collection to engagement service
□ Set up feedback endpoints for user labeling
□ Export data weekly for training

"""

