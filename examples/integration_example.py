"""
Example: How to Add Data Collection to Your API Endpoints

This file shows exactly how to instrument your endpoints to collect training data.
Copy these patterns into your actual route files.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import time

# Import the data collector
from app.train.pipelines.data_collection_pipeline import get_data_collector

# Import your services
from app.services.command_router import get_command_router
from app.services.contextual_ability_engine import get_contextual_ability_engine

router = APIRouter()

# ============================================================================
# EXAMPLE 1: Command Routing with Data Collection
# ============================================================================

class CommandRequest(BaseModel):
    command: str
    user_id: str
    user_role: str = "general"
    context: Optional[Dict] = None
    min_confidence: float = 0.7

@router.post("/intelligence/route-command")
async def route_command_example(request: CommandRequest):
    """
    Example endpoint with full data collection.
    Copy this pattern to your actual route.
    """
    collector = get_data_collector()
    router = get_command_router()
    start_time = time.time()
    
    try:
        # 1. Get prediction from your model
        result = router.route(
            command=request.command,
            user_role=request.user_role,
            context=request.context or {},
            min_confidence=request.min_confidence
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # 2. Collect classification data
        collector.collect_classification(
            task_text=request.command,
            description=request.context.get('description') if request.context else None,
            prediction={
                'type': result.get('ability_id'),
                'complexity': result.get('complexity', 'medium'),
                'estimated_duration': result.get('duration', 30)
            },
            actual=None,  # Will be filled when user provides feedback
            feedback=None
        )
        
        # 3. Collect interaction data
        collector.collect_interaction(
            user_id=request.user_id,
            action_type="intent_classification",
            context={
                "command": request.command,
                "user_role": request.user_role,
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
        # 4. Collect error data too (important for debugging)
        collector.collect_interaction(
            user_id=request.user_id,
            action_type="intent_classification",
            context={
                "command": request.command,
                "error": str(e),
                "error_type": type(e).__name__
            },
            model_used="deberta-v3-base",
            response_time_ms=(time.time() - start_time) * 1000,
            success=False,
            feedback=None
        )
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EXAMPLE 2: Ability Generation with Data Collection
# ============================================================================

class AbilityGenerationRequest(BaseModel):
    user_id: str
    user_command: str
    user_profile: Dict
    project_context: Optional[Dict] = None

@router.post("/intelligence/generate-ability")
async def generate_ability_example(request: AbilityGenerationRequest):
    """
    Example ability generation endpoint with data collection.
    """
    collector = get_data_collector()
    engine = get_contextual_ability_engine()
    start_time = time.time()
    
    try:
        # 1. Generate ability
        result = engine.generate_ability(
            user_id=request.user_id,
            user_command=request.user_command,
            user_profile=request.user_profile,
            project_context=request.project_context
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # 2. Collect ability generation data
        if result.get('success') and result.get('ability'):
            ability = result.get('ability', {})
            collector.collect_challenge_generation(
                task_text=request.user_command,
                challenge={
                    "description": ability.get('description', ''),
                    "type": ability.get('category', 'custom'),
                    "difficulty": ability.get('complexity', 'medium'),
                    "pointsReward": ability.get('momentum_cost', 0)
                },
                user_engagement=None,  # Tracked later when user uses ability
                completion_rate=None,
                performance_score=None
            )
        
        # 3. Collect interaction
        collector.collect_interaction(
            user_id=request.user_id,
            action_type="ability_generation",
            context={
                "command": request.user_command,
                "generated_ability": result.get('ability'),
                "user_profile": request.user_profile,
                "success": result.get('success', False)
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
            action_type="ability_generation",
            context={
                "command": request.user_command,
                "error": str(e)
            },
            model_used=engine.llm_model_name,
            response_time_ms=(time.time() - start_time) * 1000,
            success=False,
            feedback=None
        )
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EXAMPLE 3: Feedback Collection Endpoint
# ============================================================================

class FeedbackRequest(BaseModel):
    user_id: str
    feedback_type: str  # "classification" or "ability_generation"
    original_command: str
    original_prediction: Optional[Dict] = None
    generated_ability: Optional[Dict] = None
    correct_ability_id: Optional[str] = None
    actual_complexity: Optional[str] = None
    actual_duration: Optional[int] = None
    rating: float  # 1-5 scale
    engagement_score: Optional[float] = None  # 0-1
    ability_used: Optional[bool] = None
    performance_score: Optional[float] = None  # 0-1
    model_used: str
    comments: Optional[str] = None

@router.post("/intelligence/feedback")
async def collect_feedback_example(request: FeedbackRequest):
    """
    Critical endpoint for collecting user feedback.
    This is how you get labeled training data!
    """
    collector = get_data_collector()
    
    try:
        if request.feedback_type == "classification":
            # Update classification with correct label
            collector.collect_classification(
                task_text=request.original_command,
                description=None,
                prediction=request.original_prediction or {},
                actual={
                    'type': request.correct_ability_id,
                    'complexity': request.actual_complexity or 'medium',
                    'estimated_duration': request.actual_duration or 30
                },
                feedback=request.rating
            )
        
        elif request.feedback_type == "ability_generation":
            # Update ability generation with engagement metrics
            collector.collect_challenge_generation(
                task_text=request.original_command,
                challenge=request.generated_ability or {},
                user_engagement=request.engagement_score,
                completion_rate=1.0 if request.ability_used else 0.0,
                performance_score=request.performance_score
            )
        
        # Always collect interaction feedback
        collector.collect_interaction(
            user_id=request.user_id,
            action_type=f"{request.feedback_type}_feedback",
            context={
                "original_command": request.original_command,
                "rating": request.rating,
                "comments": request.comments,
                "feedback_type": request.feedback_type
            },
            model_used=request.model_used,
            response_time_ms=0,  # Not applicable for feedback
            success=request.rating >= 3,  # 3+ is considered success
            feedback=request.rating
        )
        
        return {
            "success": True,
            "message": "Feedback collected successfully",
            "data_points_collected": 1
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to collect feedback: {str(e)}")


# ============================================================================
# EXAMPLE 4: Minimal Integration (Just the Essentials)
# ============================================================================

@router.post("/intelligence/route-command-minimal")
async def route_command_minimal(request: CommandRequest):
    """
    Minimal example - just collect the prediction.
    Use this if you want to start simple.
    """
    collector = get_data_collector()
    router = get_command_router()
    
    result = router.route(
        command=request.command,
        user_role=request.user_role,
        context=request.context or {},
        min_confidence=request.min_confidence
    )
    
    # Just collect the classification - simplest possible
    collector.collect_classification(
        task_text=request.command,
        description=None,
        prediction={
            'type': result.get('ability_id'),
            'complexity': 'medium',
            'estimated_duration': 30
        },
        actual=None,  # Add feedback later
        feedback=None
    )
    
    return result


# ============================================================================
# USAGE NOTES
# ============================================================================

"""
HOW TO USE THESE EXAMPLES:

1. Copy the pattern from route_command_example() to your actual route
2. Replace get_command_router() with your actual service
3. Keep the data collection calls - they're the important part
4. Add the feedback endpoint - critical for getting labeled data

QUICK INTEGRATION CHECKLIST:

□ Add collector = get_data_collector() at the start of your endpoint
□ Add collector.collect_classification() after getting prediction
□ Add collector.collect_interaction() to track all API calls
□ Create a feedback endpoint so users can label predictions
□ Test that data is being saved to the database

VERIFY DATA COLLECTION:

Run this to check if data is being collected:
  python -c "from app.train.pipelines.data_collection_pipeline import get_data_collector; import sqlite3; c = get_data_collector(); conn = sqlite3.connect(c.db_path); print('Classifications:', conn.execute('SELECT COUNT(*) FROM task_classifications').fetchone()[0]); print('Interactions:', conn.execute('SELECT COUNT(*) FROM user_interactions').fetchone()[0])"

"""

