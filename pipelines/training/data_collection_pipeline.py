"""
Data Collection Pipeline for Deepiri Platform
Collects training data for:
- Tier 1: Intent Classification (50 predefined abilities)
- Tier 2: Role-based Ability Generation (dynamic, creative abilities)
- Tier 3: RL Productivity Optimization
- Prompt-to-Tasks Engine (main differentiator)
- Gamification System (Momentum, Streaks, Boosts, Objectives, Odysseys, Seasons)
"""
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import sqlite3
try:
    from deepiri_modelkit.logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger("helox.data_collection")


class DataCollectionPipeline:
    """Collect training data from various sources for Deepiri's AI Work OS."""
    
    def __init__(self, db_path: str = "data/collection.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize data collection database with all Deepiri-specific tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # ========================================================================
        # TIER 1: Intent Classification (50 predefined abilities)
        # ========================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS task_classifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_text TEXT NOT NULL,
                description TEXT,
                user_role TEXT,
                predicted_type TEXT,
                predicted_complexity TEXT,
                predicted_duration INTEGER,
                predicted_confidence REAL,
                actual_type TEXT,
                actual_complexity TEXT,
                actual_duration INTEGER,
                user_feedback REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ========================================================================
        # TIER 2: Role-based Ability Generation (dynamic abilities)
        # ========================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ability_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_role TEXT,
                user_command TEXT NOT NULL,
                generated_ability TEXT,
                ability_name TEXT,
                ability_category TEXT,
                ability_steps TEXT,
                ability_parameters TEXT,
                momentum_cost INTEGER,
                estimated_duration INTEGER,
                rag_context TEXT,
                model_used TEXT,
                user_engagement REAL,
                ability_used BOOLEAN,
                completion_rate REAL,
                performance_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ========================================================================
        # PROMPT-TO-TASKS ENGINE (Main Differentiator)
        # ========================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_to_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_role TEXT,
                prompt TEXT NOT NULL,
                generated_tasks TEXT,
                task_count INTEGER,
                subtasks TEXT,
                estimates TEXT,
                execution_plan TEXT,
                project_type TEXT,
                complexity TEXT,
                estimated_total_duration INTEGER,
                user_acceptance BOOLEAN,
                tasks_completed INTEGER,
                actual_duration INTEGER,
                user_feedback REAL,
                model_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ========================================================================
        # TIER 3: RL Productivity Optimization
        # ========================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rl_training_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                state_data TEXT,
                action_taken TEXT,
                reward REAL,
                next_state_data TEXT,
                done BOOLEAN,
                episode_id TEXT,
                step_number INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS productivity_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_state TEXT,
                recommended_action TEXT,
                recommendation_type TEXT,
                expected_benefit TEXT,
                user_acceptance BOOLEAN,
                actual_benefit REAL,
                reward_signal REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ========================================================================
        # GAMIFICATION SYSTEM DATA
        # ========================================================================
        
        # Objectives (Tasks with Momentum)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS objective_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                objective_id TEXT,
                title TEXT,
                description TEXT,
                momentum_reward INTEGER,
                status TEXT,
                deadline TEXT,
                completed_at TEXT,
                auto_detected BOOLEAN,
                odyssey_id TEXT,
                season_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Odysseys (Project Workflows)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odyssey_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                organization_id TEXT,
                odyssey_id TEXT,
                title TEXT,
                description TEXT,
                scale TEXT,
                status TEXT,
                objectives_count INTEGER,
                milestones_count INTEGER,
                progress_percentage REAL,
                completed_at TEXT,
                season_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Seasons (Sprint Cycles)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS season_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                organization_id TEXT,
                season_id TEXT,
                name TEXT,
                start_date TEXT,
                end_date TEXT,
                status TEXT,
                total_momentum_earned INTEGER,
                objectives_completed INTEGER,
                odysseys_completed INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Momentum (XP/Levels)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS momentum_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                momentum_amount INTEGER,
                source TEXT,
                source_type TEXT,
                total_momentum INTEGER,
                current_level INTEGER,
                skill_category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Streaks (Consistency Tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS streak_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                streak_type TEXT,
                streak_value INTEGER,
                action TEXT,
                cashed_in BOOLEAN,
                boost_credits_earned INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Boosts (Power-ups)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS boost_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                boost_type TEXT,
                boost_source TEXT,
                credits_used INTEGER,
                duration_minutes INTEGER,
                effectiveness_score REAL,
                tasks_completed INTEGER,
                time_saved_minutes INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ========================================================================
        # GENERAL USER INTERACTIONS
        # ========================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                user_role TEXT,
                action_type TEXT,
                context TEXT,
                model_used TEXT,
                response_time_ms REAL,
                success BOOLEAN,
                feedback REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Legacy table for backward compatibility
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS challenge_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT,
                task_text TEXT NOT NULL,
                generated_challenge TEXT,
                challenge_type TEXT,
                difficulty TEXT,
                points_reward INTEGER,
                user_engagement REAL,
                completion_rate REAL,
                performance_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Deepiri data collection database initialized with all tables")
    
    def collect_classification(
        self,
        task_text: str,
        description: Optional[str],
        prediction: Dict,
        actual: Optional[Dict] = None,
        feedback: Optional[float] = None,
        user_role: Optional[str] = None
    ):
        """Collect Tier 1: Intent classification data (50 predefined abilities)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO task_classifications 
            (task_text, description, user_role, predicted_type, predicted_complexity, 
             predicted_duration, predicted_confidence, actual_type, actual_complexity, 
             actual_duration, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_text,
            description,
            user_role,
            prediction.get('type'),
            prediction.get('complexity', 'medium'),
            prediction.get('estimated_duration', 30),
            prediction.get('confidence', 0.0),
            actual.get('type') if actual else None,
            actual.get('complexity') if actual else None,
            actual.get('estimated_duration') if actual else None,
            feedback
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Classification data collected")
    
    def collect_ability_generation(
        self,
        user_id: str,
        user_role: str,
        user_command: str,
        generated_ability: Dict,
        rag_context: Optional[Dict] = None,
        model_used: str = "gpt-4",
        user_engagement: Optional[float] = None,
        ability_used: Optional[bool] = None,
        completion_rate: Optional[float] = None,
        performance_score: Optional[float] = None
    ):
        """Collect Tier 2: Role-based ability generation data (dynamic abilities)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO ability_generations
            (user_id, user_role, user_command, generated_ability, ability_name,
             ability_category, ability_steps, ability_parameters, momentum_cost,
             estimated_duration, rag_context, model_used, user_engagement,
             ability_used, completion_rate, performance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            user_role,
            user_command,
            json.dumps(generated_ability),
            generated_ability.get('ability_name'),
            generated_ability.get('category'),
            json.dumps(generated_ability.get('steps', [])),
            json.dumps(generated_ability.get('parameters', {})),
            generated_ability.get('momentum_cost', 0),
            generated_ability.get('estimated_duration', 30),
            json.dumps(rag_context) if rag_context else None,
            model_used,
            user_engagement,
            ability_used,
            completion_rate,
            performance_score
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Ability generation data collected")
    
    def collect_challenge_generation(
        self,
        task_text: str,
        challenge: Dict,
        user_engagement: Optional[float] = None,
        completion_rate: Optional[float] = None,
        performance_score: Optional[float] = None
    ):
        """Legacy method for challenge generation (backward compatibility)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO challenge_generations
            (task_text, generated_challenge, challenge_type, difficulty, points_reward,
             user_engagement, completion_rate, performance_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_text,
            json.dumps(challenge),
            challenge.get('type'),
            challenge.get('difficulty'),
            challenge.get('pointsReward'),
            user_engagement,
            completion_rate,
            performance_score
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Challenge generation data collected")
    
    def collect_prompt_to_tasks(
        self,
        user_id: str,
        user_role: str,
        prompt: str,
        generated_tasks: List[Dict],
        execution_plan: Optional[Dict] = None,
        project_type: Optional[str] = None,
        model_used: str = "gpt-4",
        user_acceptance: Optional[bool] = None,
        tasks_completed: Optional[int] = None,
        actual_duration: Optional[int] = None,
        user_feedback: Optional[float] = None
    ):
        """Collect Prompt-to-Tasks Engine data (main differentiator)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        subtasks = [task.get('subtasks', []) for task in generated_tasks]
        estimates = [task.get('estimate', {}) for task in generated_tasks]
        
        cursor.execute("""
            INSERT INTO prompt_to_tasks
            (user_id, user_role, prompt, generated_tasks, task_count, subtasks,
             estimates, execution_plan, project_type, complexity, estimated_total_duration,
             user_acceptance, tasks_completed, actual_duration, user_feedback, model_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            user_role,
            prompt,
            json.dumps(generated_tasks),
            len(generated_tasks),
            json.dumps(subtasks),
            json.dumps(estimates),
            json.dumps(execution_plan) if execution_plan else None,
            project_type,
            execution_plan.get('complexity', 'medium') if execution_plan else 'medium',
            execution_plan.get('estimated_total_duration', 0) if execution_plan else 0,
            user_acceptance,
            tasks_completed,
            actual_duration,
            user_feedback,
            model_used
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Prompt-to-tasks data collected")
    
    def collect_rl_sequence(
        self,
        user_id: str,
        state_data: Dict,
        action_taken: str,
        reward: float,
        next_state_data: Optional[Dict] = None,
        done: bool = False,
        episode_id: Optional[str] = None,
        step_number: int = 0
    ):
        """Collect Tier 3: RL training sequence data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO rl_training_sequences
            (user_id, state_data, action_taken, reward, next_state_data, done, episode_id, step_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            json.dumps(state_data),
            action_taken,
            reward,
            json.dumps(next_state_data) if next_state_data else None,
            done,
            episode_id,
            step_number
        ))
        
        conn.commit()
        conn.close()
        logger.debug("RL sequence data collected")
    
    def collect_productivity_recommendation(
        self,
        user_id: str,
        user_state: Dict,
        recommended_action: str,
        recommendation_type: str,
        expected_benefit: Dict,
        user_acceptance: Optional[bool] = None,
        actual_benefit: Optional[float] = None,
        reward_signal: Optional[float] = None
    ):
        """Collect Tier 3: Productivity recommendation data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO productivity_recommendations
            (user_id, user_state, recommended_action, recommendation_type,
             expected_benefit, user_acceptance, actual_benefit, reward_signal)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            json.dumps(user_state),
            recommended_action,
            recommendation_type,
            json.dumps(expected_benefit),
            user_acceptance,
            actual_benefit,
            reward_signal
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Productivity recommendation data collected")
    
    def collect_objective_data(
        self,
        user_id: str,
        objective_id: str,
        title: str,
        momentum_reward: int,
        status: str,
        description: Optional[str] = None,
        deadline: Optional[str] = None,
        completed_at: Optional[str] = None,
        auto_detected: bool = False,
        odyssey_id: Optional[str] = None,
        season_id: Optional[str] = None
    ):
        """Collect Objective (task with momentum) data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO objective_data
            (user_id, objective_id, title, description, momentum_reward, status,
             deadline, completed_at, auto_detected, odyssey_id, season_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            objective_id,
            title,
            description,
            momentum_reward,
            status,
            deadline,
            completed_at,
            auto_detected,
            odyssey_id,
            season_id
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Objective data collected")
    
    def collect_odyssey_data(
        self,
        user_id: str,
        odyssey_id: str,
        title: str,
        scale: str,
        status: str,
        description: Optional[str] = None,
        organization_id: Optional[str] = None,
        objectives_count: int = 0,
        milestones_count: int = 0,
        progress_percentage: float = 0.0,
        completed_at: Optional[str] = None,
        season_id: Optional[str] = None
    ):
        """Collect Odyssey (project workflow) data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO odyssey_data
            (user_id, organization_id, odyssey_id, title, description, scale, status,
             objectives_count, milestones_count, progress_percentage, completed_at, season_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            organization_id,
            odyssey_id,
            title,
            description,
            scale,
            status,
            objectives_count,
            milestones_count,
            progress_percentage,
            completed_at,
            season_id
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Odyssey data collected")
    
    def collect_season_data(
        self,
        user_id: str,
        season_id: str,
        name: str,
        start_date: str,
        end_date: str,
        status: str,
        organization_id: Optional[str] = None,
        total_momentum_earned: int = 0,
        objectives_completed: int = 0,
        odysseys_completed: int = 0
    ):
        """Collect Season (sprint cycle) data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO season_data
            (user_id, organization_id, season_id, name, start_date, end_date, status,
             total_momentum_earned, objectives_completed, odysseys_completed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            organization_id,
            season_id,
            name,
            start_date,
            end_date,
            status,
            total_momentum_earned,
            objectives_completed,
            odysseys_completed
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Season data collected")
    
    def collect_momentum_event(
        self,
        user_id: str,
        momentum_amount: int,
        source: str,
        source_type: str,
        total_momentum: int,
        current_level: int,
        skill_category: Optional[str] = None
    ):
        """Collect Momentum (XP) event data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO momentum_events
            (user_id, momentum_amount, source, source_type, total_momentum,
             current_level, skill_category)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            momentum_amount,
            source,
            source_type,
            total_momentum,
            current_level,
            skill_category
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Momentum event collected")
    
    def collect_streak_event(
        self,
        user_id: str,
        streak_type: str,
        streak_value: int,
        action: str,
        cashed_in: bool = False,
        boost_credits_earned: int = 0
    ):
        """Collect Streak (consistency) event data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO streak_events
            (user_id, streak_type, streak_value, action, cashed_in, boost_credits_earned)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            streak_type,
            streak_value,
            action,
            cashed_in,
            boost_credits_earned
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Streak event collected")
    
    def collect_boost_usage(
        self,
        user_id: str,
        boost_type: str,
        boost_source: str,
        credits_used: int,
        duration_minutes: int,
        effectiveness_score: Optional[float] = None,
        tasks_completed: Optional[int] = None,
        time_saved_minutes: Optional[int] = None
    ):
        """Collect Boost (power-up) usage data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO boost_usage
            (user_id, boost_type, boost_source, credits_used, duration_minutes,
             effectiveness_score, tasks_completed, time_saved_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            boost_type,
            boost_source,
            credits_used,
            duration_minutes,
            effectiveness_score,
            tasks_completed,
            time_saved_minutes
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Boost usage collected")
    
    def collect_interaction(
        self,
        user_id: str,
        action_type: str,
        context: Dict,
        model_used: str,
        response_time_ms: float,
        success: bool,
        feedback: Optional[float] = None,
        user_role: Optional[str] = None
    ):
        """Collect general user interaction data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_interactions
            (user_id, user_role, action_type, context, model_used, response_time_ms, success, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            user_role,
            action_type,
            json.dumps(context),
            model_used,
            response_time_ms,
            success,
            feedback
        ))
        
        conn.commit()
        conn.close()
        logger.debug("Interaction data collected")
    
    def export_for_training(self, output_path: str, data_type: str = "classification"):
        """Export collected data for training."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if data_type == "classification":
            # Tier 1: Intent Classification
            cursor.execute("""
                SELECT task_text, description, user_role, predicted_type, predicted_complexity,
                       predicted_confidence, actual_type, actual_complexity, user_feedback
                FROM task_classifications
                WHERE actual_type IS NOT NULL OR user_feedback IS NOT NULL
            """)
            
            with open(output_path, 'w') as f:
                for row in cursor.fetchall():
                    data = {
                        "text": row[0] + (" " + row[1] if row[1] else ""),
                        "label": row[6] if row[6] else row[3],
                        "metadata": {
                            "user_role": row[2],
                            "complexity": row[7] if row[7] else row[4],
                            "confidence": row[5],
                            "feedback": row[8]
                        }
                    }
                    f.write(json.dumps(data) + '\n')
        
        elif data_type == "ability_generation":
            # Tier 2: Role-based Ability Generation
            cursor.execute("""
                SELECT user_id, user_role, user_command, generated_ability, ability_name,
                       ability_category, rag_context, model_used, user_engagement,
                       ability_used, completion_rate, performance_score
                FROM ability_generations
                WHERE user_engagement IS NOT NULL OR ability_used IS NOT NULL
            """)
            
            with open(output_path, 'w') as f:
                for row in cursor.fetchall():
                    ability = json.loads(row[3])
                    data = {
                        "input": row[2],
                        "output": ability,
                        "metadata": {
                            "user_id": row[0],
                            "user_role": row[1],
                            "ability_name": row[4],
                            "category": row[5],
                            "rag_context": json.loads(row[6]) if row[6] else None,
                            "model_used": row[7],
                            "engagement": row[8],
                            "ability_used": row[9],
                            "completion_rate": row[10],
                            "performance": row[11]
                        }
                    }
                    f.write(json.dumps(data) + '\n')
        
        elif data_type == "prompt_to_tasks":
            # Prompt-to-Tasks Engine
            cursor.execute("""
                SELECT user_id, user_role, prompt, generated_tasks, execution_plan,
                       project_type, user_acceptance, tasks_completed, actual_duration, user_feedback
                FROM prompt_to_tasks
                WHERE user_acceptance IS NOT NULL
            """)
            
            with open(output_path, 'w') as f:
                for row in cursor.fetchall():
                    data = {
                        "input": row[2],
                        "output": json.loads(row[3]),
                        "metadata": {
                            "user_id": row[0],
                            "user_role": row[1],
                            "execution_plan": json.loads(row[4]) if row[4] else None,
                            "project_type": row[5],
                            "user_acceptance": row[6],
                            "tasks_completed": row[7],
                            "actual_duration": row[8],
                            "feedback": row[9]
                        }
                    }
                    f.write(json.dumps(data) + '\n')
        
        elif data_type == "rl_training":
            # Tier 3: RL Training Sequences
            cursor.execute("""
                SELECT user_id, state_data, action_taken, reward, next_state_data,
                       done, episode_id, step_number
                FROM rl_training_sequences
                ORDER BY episode_id, step_number
            """)
            
            with open(output_path, 'w') as f:
                for row in cursor.fetchall():
                    data = {
                        "state": json.loads(row[1]),
                        "action": row[2],
                        "reward": row[3],
                        "next_state": json.loads(row[4]) if row[4] else None,
                        "done": bool(row[5]),
                        "metadata": {
                            "user_id": row[0],
                            "episode_id": row[6],
                            "step_number": row[7]
                        }
                    }
                    f.write(json.dumps(data) + '\n')
        
        elif data_type == "challenge":
            # Legacy challenge generation
            cursor.execute("""
                SELECT task_text, generated_challenge, challenge_type, difficulty,
                       user_engagement, completion_rate, performance_score
                FROM challenge_generations
                WHERE user_engagement IS NOT NULL
            """)
            
            with open(output_path, 'w') as f:
                for row in cursor.fetchall():
                    challenge = json.loads(row[1])
                    data = {
                        "input": row[0],
                        "output": challenge.get('description', ''),
                        "metadata": {
                            "type": row[2],
                            "difficulty": row[3],
                            "engagement": row[4],
                            "completion_rate": row[5],
                            "performance": row[6]
                        }
                    }
                    f.write(json.dumps(data) + '\n')
        
        elif data_type == "gamification":
            # Export all gamification data
            gamification_data = {
                "objectives": [],
                "odysseys": [],
                "seasons": [],
                "momentum_events": [],
                "streak_events": [],
                "boost_usage": []
            }
            
            # Objectives
            cursor.execute("SELECT * FROM objective_data")
            for row in cursor.fetchall():
                gamification_data["objectives"].append({
                    "user_id": row[1],
                    "objective_id": row[2],
                    "title": row[3],
                    "momentum_reward": row[5],
                    "status": row[6],
                    "completed_at": row[8]
                })
            
            # Momentum events
            cursor.execute("SELECT * FROM momentum_events")
            for row in cursor.fetchall():
                gamification_data["momentum_events"].append({
                    "user_id": row[1],
                    "momentum_amount": row[2],
                    "source": row[3],
                    "source_type": row[4],
                    "skill_category": row[7]
                })
            
            # Boost usage
            cursor.execute("SELECT * FROM boost_usage")
            for row in cursor.fetchall():
                gamification_data["boost_usage"].append({
                    "user_id": row[1],
                    "boost_type": row[2],
                    "effectiveness_score": row[6],
                    "tasks_completed": row[7],
                    "time_saved_minutes": row[8]
                })
            
            with open(output_path, 'w') as f:
                f.write(json.dumps(gamification_data, indent=2))
        
        conn.close()
        logger.info("Data exported for training", path=output_path, type=data_type)


_data_collector = None

def get_data_collector() -> DataCollectionPipeline:
    """Get singleton data collector."""
    global _data_collector
    if _data_collector is None:
        _data_collector = DataCollectionPipeline()
    return _data_collector


