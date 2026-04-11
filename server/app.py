"""Email Triage Environment - FastAPI Server"""
import json
import random
import os
import sys
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EmailObservation, EmailAction, EmailReward, StepResult


class EmailTriageEnv:
    """Email triage environment with 3 difficulty levels (easy, medium, hard)"""

    def __init__(self):
        # Load tasks dataset - try multiple paths for compatibility
        possible_paths = [
            '/app/tasks.json',  # Docker container (first priority)
            os.path.join(os.path.dirname(__file__), '..', 'tasks.json'),  # Relative to this file
            os.path.join(os.getcwd(), 'tasks.json'),  # Current working directory
            './tasks.json',  # Current directory
            'tasks.json',  # Relative to pwd
        ]
        
        tasks_path = None
        print(f"[INIT] Searching for tasks.json...", flush=True)
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            exists = os.path.exists(path)
            print(f"[INIT]   Checking {abs_path}: {exists}", flush=True)
            if exists:
                tasks_path = path
                print(f"[INIT] ✓ Found tasks.json at: {os.path.abspath(path)}", flush=True)
                break
        
        if tasks_path is None:
            error_msg = (
                f"tasks.json not found. Tried:\n"
                f"  {chr(10).join(f'  - {os.path.abspath(p)}' for p in possible_paths)}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Script directory: {os.path.dirname(__file__)}"
            )
            raise FileNotFoundError(error_msg)
        
        print(f"[INIT] Loading tasks from {os.path.abspath(tasks_path)}...", flush=True)
        with open(tasks_path, 'r') as f:
            self.tasks = json.load(f)
        
        print(f"[INIT] ✓ Loaded {len(self.tasks)} tasks", flush=True)

        # Group by difficulty
        self.tasks_by_difficulty = {
            'easy': [t for t in self.tasks if t['difficulty'] == 'easy'],
            'medium': [t for t in self.tasks if t['difficulty'] == 'medium'],
            'hard': [t for t in self.tasks if t['difficulty'] == 'hard'],
        }
        
        print(f"✓ Task distribution - easy: {len(self.tasks_by_difficulty['easy'])}, "
              f"medium: {len(self.tasks_by_difficulty['medium'])}, "
              f"hard: {len(self.tasks_by_difficulty['hard'])}")

        self.state = None
        self.current_task = None
        self.current_observation = None  # store for step() to return

    def reset(self) -> EmailObservation:
        """Reset environment and return initial observation"""
        # Pick random difficulty for this episode
        difficulty = random.choice(['easy', 'medium', 'hard'])
        tasks_for_difficulty = self.tasks_by_difficulty[difficulty]
        self.current_task = random.choice(tasks_for_difficulty)

        self.state = {
            'task_id': self.current_task['task_id'],
            'difficulty': difficulty,
            'gold_label': self.current_task['gold_label'],
            'rubric': self.current_task.get('rubric', {}),
            'step_count': 0,
            'done': False,
        }

        observation = EmailObservation(
            task_id=self.current_task['task_id'],
            subject=self.current_task['subject'],
            sender=self.current_task['sender'],
            body=self.current_task['body'],
            difficulty=difficulty,
            thread_history=self.current_task.get('thread_history', []),
            timestamp=self.current_task.get('timestamp', '2025-04-08T00:00:00Z')
        )
        self.current_observation = observation  # cache for step() to return
        return observation

    def step(self, action: EmailAction) -> EmailReward:
        """Process action and return reward"""
        if self.state is None:
            raise ValueError("Must call reset() before step()")

        # Score label accuracy
        label_score = 1.0 if action.label == self.state['gold_label'] else 0.0

        # Score reply quality (if provided and needed)
        reply_score = 0.0
        if action.draft_reply and self.state['gold_label'] == 'needs_reply':
            reply_score = self.score_reply(action.draft_reply, self.state['rubric'])
        elif not action.draft_reply and self.state['gold_label'] == 'needs_reply':
            reply_score = 0.0  # Penalize missing reply
        elif action.draft_reply and self.state['gold_label'] != 'needs_reply':
            reply_score = 0.2  # Penalize unnecessary reply

        # Combined reward: 60% label + 40% reply
        reward = 0.6 * label_score + 0.4 * reply_score
        reward = round(reward, 2)

        self.state['step_count'] += 1
        self.state['done'] = True

        # OpenEnv spec: step() must return observation + reward + done + info
        return StepResult(
            observation=self.current_observation,
            reward=reward,
            done=True,
            info={
                'task_id': self.state['task_id'],
                'difficulty': self.state['difficulty'],
                'gold_label': self.state['gold_label'],
                'predicted_label': action.label,
            },
            label_score=round(label_score, 2),
            reply_score=round(reply_score, 2),
        )

    def score_reply(self, draft_reply: str, rubric: dict) -> float:
        """
        Score reply quality using heuristic-based evaluation.
        Checks: tone, required keywords, and length constraints.
        
        Returns: Float between 0.0 and 1.0
        """
        score_components = []
        
        # 1. Length check (25%)
        words = len(draft_reply.split())
        max_words = rubric.get('max_words', 100)
        if words <= max_words:
            length_score = min(1.0, words / (max_words * 0.8))  # Reward 80% of max
        else:
            length_score = max(0.0, 1.0 - (words - max_words) / max_words)
        score_components.append(length_score * 0.25)
        
        # 2. Must-include check (50%)
        must_include = rubric.get('must_include', [])
        if must_include:
            reply_lower = draft_reply.lower()
            found_count = sum(1 for phrase in must_include if phrase.lower() in reply_lower)
            include_score = found_count / len(must_include) if must_include else 1.0
            score_components.append(include_score * 0.50)
        else:
            score_components.append(0.50)
        
        # 3. Keywords check (25%)
        keywords = rubric.get('keywords', [])
        if keywords:
            reply_lower = draft_reply.lower()
            found_count = sum(1 for keyword in keywords if keyword.lower() in reply_lower)
            keyword_score = found_count / len(keywords) if keywords else 1.0
            score_components.append(keyword_score * 0.25)
        else:
            score_components.append(0.25)
        
        # 4. Tone consistency (implicit from rubric)
        tone = rubric.get('tone', '').lower()
        if 'formal' in tone and any(word in draft_reply.lower() for word in ['please', 'thank', 'appreciate', 'regards']):
            score_components.append(0.10)
        elif 'casual' in tone and any(word in draft_reply.lower() for word in ['hi', 'hey', 'cheers', 'thanks']):
            score_components.append(0.10)
        else:
            score_components.append(0.05)
        
        # Combine all components  (normalized to 0-1)
        final_score = sum(score_components) 
        return round(min(1.0, max(0.0, final_score)), 2)

    def get_state(self) -> dict:
        """Return current state (excluding gold labels)"""
        if self.state is None:
            return {'error': 'Environment not initialized. Call reset() first.'}
        return {
            'task_id': self.state['task_id'],
            'difficulty': self.state['difficulty'],
            'step_count': self.state['step_count'],
            'done': self.state['done'],
        }


# Global environment instance - initialized on startup
env = None
init_error = None
init_time = None

def initialize_env():
    """Initialize environment with error handling"""
    global env, init_error, init_time
    import time
    start = time.time()
    try:
        print("[INIT] Starting EmailTriageEnv initialization...", flush=True)
        print(f"[INIT] Current working directory: {os.getcwd()}", flush=True)
        print(f"[INIT] __file__ = {__file__}", flush=True)
        
        env = EmailTriageEnv()
        
        init_time = time.time() - start
        print(f"✓ Environment initialized in {init_time:.2f}s", flush=True)
        return True
    except Exception as e:
        init_error = str(e)
        init_time = time.time() - start
        print(f"✗ Environment init failed after {init_time:.2f}s", file=sys.stderr, flush=True)
        print(f"✗ Exception: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return False

# FastAPI app - FastAPI must be created first
app = FastAPI(
    title="Email Triage OpenEnv",
    description="Real-world email classification and reply generation environment",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize on module load (BLOCKING - prevents uvicorn from serving until ready)
print("[MODULE_LOAD] Initializing environment on import (BLOCKING)...", flush=True)
success = initialize_env()
if not success:
    print(f"[MODULE_LOAD] ✗ FAILED: {init_error}", file=sys.stderr, flush=True)
    # Still allow server to start but all endpoints will fail gracefully
else:
    print("[MODULE_LOAD] ✓ Environment ready before server binding", flush=True)


@app.on_event("startup")
async def startup():
    """Startup event - validate environment is initialized"""
    global env, init_error
    print("[STARTUP] APP startup event", flush=True)
    if env is None:
        print("[STARTUP] WARNING: env is None, retrying initialization", file=sys.stderr, flush=True)
        initialize_env()
    print(f"[STARTUP] Environment status: {'ready' if env is not None else 'NOT READY'}", flush=True)


@app.get("/")
async def root():
    """Root endpoint - simple and fast"""
    if env is None:
        return {"status": "initializing", "error": init_error}, 503
    return {"status": "ok", "version": "1.0.0", "tasks": len(env.tasks)}


@app.get("/health")
async def health():
    """Liveness probe - must respond instantly"""
    return {"status": "alive"}


@app.get("/readiness")
async def readiness():
    """Readiness probe - checks if environment is ready"""
    if env is None:
        return {"status": "initializing", "error": init_error}, 503
    return {"status": "ready", "tasks": len(env.tasks)}


@app.get("/debug")
async def debug():
    """Debug endpoint - shows initialization status and file paths"""
    return {
        "env_initialized": env is not None,
        "task_count": len(env.tasks) if env else 0,
        "init_error": init_error,
        "init_time": init_time,
        "cwd": os.getcwd(),
        "script_dir": os.path.dirname(__file__),
        "tasks_file_exists": os.path.exists(os.path.join(os.getcwd(), 'tasks.json')),
        "app_tasks_file_exists": os.path.exists('/app/tasks.json'),
    }


@app.get("/api/config")
async def api_config():
    """OpenEnv API configuration"""
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    return {
        "name": "email_triage_agent",
        "version": "1.0.0",
        "description": "Real-world email triage and smart reply environment",
        "endpoints": {
            "reset": "/reset",
            "step": "/step",
            "state": "/state"
        },
        "tasks": {
            "easy": len(env.tasks_by_difficulty.get("easy", [])),
            "medium": len(env.tasks_by_difficulty.get("medium", [])),
            "hard": len(env.tasks_by_difficulty.get("hard", []))
        }
    }


@app.post("/reset")
async def reset():
    """Reset environment and return initial observation"""
    if env is None:
        print(f"[/reset] ERROR: env is None, init_error={init_error}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    try:
        print("[/reset] Calling env.reset()", flush=True)
        observation = env.reset()
        result = observation.model_dump()
        print(f"[/reset] SUCCESS: returning observation", flush=True)
        return result
    except Exception as e:
        print(f"[/reset] ERROR: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(action: EmailAction):
    """Process action and return reward"""
    if env is None:
        raise HTTPException(status_code=500, detail=f"Environment not initialized: {init_error}")
    try:
        reward_data = env.step(action)
        return reward_data.model_dump()
    except Exception as e:
        print(f"✗ Error in /step: {e}", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Return current environment state"""
    if env is None:
        raise HTTPException(status_code=500, detail=f"Environment not initialized: {init_error}")
    try:
        return env.get_state()
    except Exception as e:
        print(f"✗ Error in /state: {e}", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
