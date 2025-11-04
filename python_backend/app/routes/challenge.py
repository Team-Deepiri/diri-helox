from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import openai
from ..settings import settings
from ..logging_config import get_logger, ErrorLogger
import time
import json
import asyncio

router = APIRouter()
logger = get_logger("pyagent.challenge")
error_logger = ErrorLogger()


class TaskInput(BaseModel):
    title: str
    description: Optional[str] = None
    type: Optional[str] = "manual"
    estimatedDuration: Optional[int] = None


class ChallengeGenerateRequest(BaseModel):
    task: TaskInput


class ChallengeResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    request_id: str


@router.post("/challenge/generate", response_model=ChallengeResponse)
async def generate_challenge(req: ChallengeGenerateRequest, request: Request):
    """
    Generate a gamified challenge from a task using AI.
    
    Args:
        req: Challenge generation request with task data
        request: FastAPI request object for logging
        
    Returns:
        ChallengeResponse with generated challenge data
    """
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    if not settings.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not configured", request_id=request_id)
        raise HTTPException(status_code=503, detail="AI service not configured")

    logger.info("Generating challenge from task", 
                request_id=request_id, 
                task_title=req.task.title)

    try:
        start_time = time.time()
        client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Determine challenge type based on task type
        challenge_type_mapping = {
            'study': 'quiz',
            'code': 'coding_challenge',
            'creative': 'puzzle',
            'manual': 'timed_completion'
        }
        default_challenge_type = challenge_type_mapping.get(req.task.type, 'timed_completion')
        
        # Create prompt for challenge generation
        system_prompt = """You are Deepiri AI Challenge Generator. Your job is to convert tasks into engaging, gamified challenges that motivate users to complete their work.

Generate a challenge that:
1. Makes the task fun and engaging
2. Has appropriate difficulty based on task complexity
3. Includes clear instructions
4. Has a reasonable time limit if applicable
5. Rewards completion with points

Return a JSON object with:
- type: one of ['quiz', 'puzzle', 'coding_challenge', 'timed_completion', 'streak']
- title: engaging challenge title
- description: clear instructions
- difficulty: 'easy', 'medium', 'hard', or 'adaptive'
- difficultyScore: 1-10
- pointsReward: base points (typically 100-500)
- configuration: challenge-specific settings (timeLimit, questions, hints, etc.)
"""

        user_prompt = f"""Task: {req.task.title}
Description: {req.task.description or 'No description'}
Type: {req.task.type}
Estimated Duration: {req.task.estimatedDuration or 'Unknown'} minutes

Generate a {default_challenge_type} challenge for this task. Make it engaging and fun!"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        completion_params = {
            "model": settings.OPENAI_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }

        completion = await asyncio.to_thread(
            client.chat.completions.create,
            **completion_params
        )
        
        generation_time = time.time() - start_time

        # Parse JSON response
        response_content = completion.choices[0].message.content
        challenge_data = json.loads(response_content)

        # Ensure required fields
        challenge_data.setdefault('type', default_challenge_type)
        challenge_data.setdefault('title', f"Challenge: {req.task.title}")
        challenge_data.setdefault('description', f"Complete this task: {req.task.title}")
        challenge_data.setdefault('difficulty', 'medium')
        challenge_data.setdefault('difficultyScore', 5)
        challenge_data.setdefault('pointsReward', 100)
        challenge_data.setdefault('configuration', {})

        # Add metadata
        challenge_data['aiMetadata'] = {
            'model': completion.model,
            'prompt': system_prompt,
            'generationTime': generation_time
        }

        logger.info("Challenge generated successfully",
                   request_id=request_id,
                   challenge_type=challenge_data['type'],
                   generation_time_ms=generation_time * 1000)

        return ChallengeResponse(
            success=True,
            data=challenge_data,
            request_id=request_id
        )

    except json.JSONDecodeError as e:
        logger.error("Failed to parse AI challenge response", request_id=request_id, error=str(e))
        # Return default challenge on parse error
        return ChallengeResponse(
            success=True,
            data={
                'type': default_challenge_type,
                'title': f"Complete: {req.task.title}",
                'description': f"Complete this task within {req.task.estimatedDuration or 30} minutes!",
                'difficulty': 'medium',
                'difficultyScore': 5,
                'pointsReward': 100,
                'configuration': {
                    'timeLimit': req.task.estimatedDuration or 30
                },
                'aiMetadata': {
                    'model': 'fallback',
                    'generationTime': 0
                }
            },
            request_id=request_id
        )
    
    except openai.RateLimitError as e:
        logger.error("OpenAI rate limit exceeded", request_id=request_id, error=str(e))
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    
    except openai.AuthenticationError as e:
        logger.error("OpenAI authentication failed", request_id=request_id, error=str(e))
        raise HTTPException(status_code=503, detail="AI service authentication failed")
    
    except openai.APIError as e:
        logger.error("OpenAI API error", request_id=request_id, error=str(e))
        raise HTTPException(status_code=502, detail="AI service temporarily unavailable")
    
    except Exception as e:
        error_logger.log_api_error(e, request_id, "/agent/challenge/generate")
        logger.error("Unexpected error in challenge generation", request_id=request_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


