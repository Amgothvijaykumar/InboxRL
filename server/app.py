"""Email Triage Environment - FastAPI Server"""
import json
import random
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models import EmailObservation, EmailAction, EmailReward


class EmailTriageEnv:
    """Email triage environment with 3 difficulty levels (easy, medium, hard)"""

    def __init__(self):
        # Load tasks dataset
        tasks_path = os.path.join(os.path.dirname(__file__), '..', 'tasks.json')
        with open(tasks_path, 'r') as f:
            self.tasks = json.load(f)

        # Group by difficulty
        self.tasks_by_difficulty = {
            'easy': [t for t in self.tasks if t['difficulty'] == 'easy'],
            'medium': [t for t in self.tasks if t['difficulty'] == 'medium'],
            'hard': [t for t in self.tasks if t['difficulty'] == 'hard'],
        }

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
            difficulty=difficulty
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
            # Simple heuristic: reward length and acknowledgment
            words = len(action.draft_reply.split())
            has_ack = any(word.lower() in action.draft_reply.lower() 
                         for word in ['acknowledge', 'received', 'thank', 'appreciate'])
            reply_score = min(1.0, (words / 50.0) * 0.5 + (0.5 if has_ack else 0.0))
        elif not action.draft_reply and self.state['gold_label'] == 'needs_reply':
            reply_score = 0.0  # Penalize missing reply
        elif action.draft_reply and self.state['gold_label'] != 'needs_reply':
            reply_score = 0.3  # Penalize unnecessary reply

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


# Global environment instance
env = EmailTriageEnv()

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
    return {
        "name": "Email Triage OpenEnv",
        "version": "1.0.0",
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
        }
    }


@app.post("/reset")
async def reset():
    """Reset environment and return initial observation"""
    try:
        observation = env.reset()
        return observation.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action: EmailAction):
    """Process action and return reward"""
    try:
        reward_data = env.step(action)
        return reward_data.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Return current environment state"""
    try:
        return env.get_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
