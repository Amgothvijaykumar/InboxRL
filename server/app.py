"""Email Triage Environment - FastAPI Server"""
import json
import random
import os
import sys
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import EmailObservation, EmailAction, EmailReward


class EmailTriageEnv:
    """Email triage environment with 3 difficulty levels (easy, medium, hard)"""

    def __init__(self):
        # Load tasks dataset - try multiple paths for compatibility
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'tasks.json'),  # Local dev
            '/app/tasks.json',  # Docker container
            './tasks.json',  # Current directory
            os.path.join(os.getcwd(), 'tasks.json'),  # CWD
        ]
        
        tasks_path = None
        for path in possible_paths:
            if os.path.exists(path):
                tasks_path = path
                print(f"✓ Found tasks.json at: {path}")
                break
        
        if tasks_path is None:
            raise FileNotFoundError(
                f"tasks.json not found. Tried: {possible_paths}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Script directory: {os.path.dirname(__file__)}"
            )
        
        with open(tasks_path, 'r') as f:
            self.tasks = json.load(f)
        
        print(f"✓ Loaded {len(self.tasks)} tasks from {tasks_path}")

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

        return EmailObservation(
            task_id=self.current_task['task_id'],
            subject=self.current_task['subject'],
            sender=self.current_task['sender'],
            body=self.current_task['body'],
            difficulty=difficulty,
            thread_history=self.current_task.get('thread_history', []),
            timestamp=self.current_task.get('timestamp', '2025-04-08T00:00:00Z')
        )

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

        return EmailReward(
            label_score=round(label_score, 2),
            reply_score=round(reply_score, 2),
            reward=reward,
            done=True,
            info={
                'task_id': self.state['task_id'],
                'difficulty': self.state['difficulty'],
                'gold_label': self.state['gold_label'],
                'predicted_label': action.label,
            }
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
        env = EmailTriageEnv()
        init_time = time.time() - start
        print(f"✓ Environment initialized in {init_time:.2f}s", flush=True)
        return True
    except Exception as e:
        init_error = str(e)
        init_time = time.time() - start
        print(f"✗ Environment init failed after {init_time:.2f}s: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return False

# Try to initialize on module load
try:
    initialize_env()
except Exception as e:
    print(f"✗ Critical error during module load: {e}", file=sys.stderr, flush=True)
    init_error = str(e)

# FastAPI app
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


@app.get("/")
async def root():
    """Root endpoint - simple and fast"""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Liveness probe - must respond instantly"""
    return {"status": "alive"}


@app.get("/readiness")
async def readiness():
    """Readiness probe - checks if environment is ready"""
    if env is None:
        return {"status": "not_ready", "error": init_error}, 503
    return {"status": "ready", "tasks": len(env.tasks)}


@app.post("/reset")
async def reset():
    """Reset environment and return initial observation"""
    if env is None:
        raise HTTPException(status_code=500, detail=f"Environment not initialized: {init_error}")
    try:
        observation = env.reset()
        return observation.model_dump()
    except Exception as e:
        print(f"✗ Error in /reset: {e}", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
