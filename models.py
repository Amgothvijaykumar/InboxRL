"""Email Triage Environment - OpenEnv Models"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class EmailObservation(BaseModel):
    """What the agent sees"""
    task_id: str = Field(..., description="Task identifier")
    subject: str = Field(..., description="Email subject")
    sender: str = Field(..., description="Sender email/name")
    body: str = Field(..., description="Email body content")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Task difficulty")


class EmailAction(BaseModel):
    """What the agent outputs"""
    label: Literal["urgent", "needs_reply", "fyi", "junk"] = Field(
        ..., description="Email classification"
    )
    draft_reply: Optional[str] = Field(
        None, description="Optional draft reply if label is needs_reply"
    )


class EmailReward(BaseModel):
    """Reward signal"""
    label_score: float = Field(..., ge=0.0, le=1.0, description="Label accuracy (0-1)")
    reply_score: float = Field(..., ge=0.0, le=1.0, description="Reply quality (0-1)")
    reward: float = Field(..., ge=0.0, le=1.0, description="Combined reward")
    done: bool = Field(..., description="Episode complete")
    info: dict = Field(default_factory=dict, description="Additional info")
