"""Email Triage Environment - OpenEnv Models"""
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field

try:
    from openm.engine.types import Action, Observation, State
except ImportError:
    # Fallback for development without openm-core
    Action = BaseModel
    Observation = BaseModel
    State = BaseModel


class EmailObservation(Observation):
    """What the agent sees: email content"""
    task_id: str = Field(..., description="Task identifier")
    subject: str = Field(..., description="Email subject")
    sender: str = Field(..., description="Sender email/name")
    body: str = Field(..., description="Email body content")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Task difficulty")
    thread_history: List[str] = Field(default_factory=list, description="Previous messages in thread")
    timestamp: str = Field(..., description="ISO format timestamp")


class EmailAction(Action):
    """Agent's output: classification + optional draft reply"""
    label: Literal["urgent", "needs_reply", "fyi", "junk"] = Field(
        ..., description="Email classification"
    )
    draft_reply: Optional[str] = Field(
        None, description="Optional draft reply if label is needs_reply"
    )


class EmailState(State):
    """Full internal environment state (hidden from agent)"""
    current_email: EmailObservation
    gold_label: str = Field(..., description="Ground truth label (hidden)")
    gold_reply_rubric: dict = Field(default_factory=dict, description="Evaluation rubric (hidden)")
    step_count: int = 0
    done: bool = False


class EmailReward(BaseModel):
    """Reward signal"""
    label_score: float = Field(..., ge=0.0, le=1.0, description="Label accuracy (0-1)")
    reply_score: float = Field(..., ge=0.0, le=1.0, description="Reply quality (0-1)")
    reward: float = Field(..., ge=0.0, le=1.0, description="Combined reward")
    done: bool = Field(..., description="Episode complete")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info")


class StepResult(BaseModel):
    """Full step() return value per OpenEnv spec: observation + reward + done + info"""
    observation: EmailObservation = Field(..., description="Current environment observation")
    reward: float = Field(..., ge=0.0, le=1.0, description="Combined reward signal")
    done: bool = Field(..., description="Episode termination flag")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    # Detailed reward breakdown (bonus fields)
    label_score: float = Field(..., ge=0.0, le=1.0, description="Label accuracy (0-1)")
    reply_score: float = Field(..., ge=0.0, le=1.0, description="Reply quality (0-1)")
