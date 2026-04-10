---
title: Email Triage Agent
emoji: 🏆
colorFrom: purple
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# Email Triage & Smart Reply Agent — OpenEnv

A real-world reinforcement learning environment for email classification, prioritization, and intelligent reply generation.


## Overview

**Email Triage** is an OpenEnv environment that simulates the critical task of managing an overflowing inbox. Agents must:
1. **Classify emails** into 4 categories: `urgent`, `needs_reply`, `fyi`, `junk`
2. **Prioritize responses** based on sender, subject, and content
3. **Draft intelligent replies** when appropriate

This environment is designed for training post-training agents to handle real-world email workloads with partial rewards and multi-trajectory learning.

## Real-World Utility

Email management is one of the most universal pain points in professional life. Modern LLMs can improve this task significantly through:
- **Inbox triage:** Automatically prioritizing what needs immediate attention
- **Draft generation:** Suggesting context-aware, professional replies
- **Follow-up tracking:** Identifying emails that require follow-up action

This environment directly targets these use cases and provides measurable, learnable rewards.

## Environment Specification

### Observation Space

```python
class EmailObservation(BaseModel):
    task_id: str              # Unique task identifier
    subject: str              # Email subject line
    sender: str               # Sender name/email
    body: str                 # Email body content
    difficulty: Literal["easy", "medium", "hard"]  # Task difficulty
```

### Action Space

```python
class EmailAction(BaseModel):
    label: Literal["urgent", "needs_reply", "fyi", "junk"]
    draft_reply: Optional[str] = None  # Required if label == "needs_reply"
```

### Reward Space

```python
class EmailReward(BaseModel):
    label_score: float        # Accuracy of classification (0-1)
    reply_score: float        # Quality of reply (0-1)
    reward: float             # Combined: 0.6*label + 0.4*reply
    done: bool                # Episode termination
    info: dict                # Additional metadata
```

## Reward Formula

```
reward = 0.6 × label_score + 0.4 × reply_score
```

### Label Score
- **1.0** if predicted label matches ground truth
- **0.0** otherwise
- Fast, deterministic evaluation

### Reply Score
- Evaluated on draft reply quality when applicable
- **Tone:** formal vs casual (from rubric)
- **Content:** must-include phrases and action items
- **Length:** bounded by max_words constraint
- **0.0** if reply missing but required
- **0.3** if reply given but not required (penalize over-replying)

## Tasks & Difficulty Levels

### Easy (2 tasks)
Classic, obvious signals:
- **Spam detection:** Free money, click-here spam
- **Critical alerts:** "CRITICAL", "DOWN", "ASAP" keywords

Expected score: 0.95+

### Medium (2 tasks)
Mixed signals, context-dependent:
- **FYI vs reply:** Messages that could go either way
- **CC'd threads:** Is recipient stakeholder or bystander?

Expected score: 0.70–0.85

### Hard (2 tasks)
Ambiguous, requires reasoning:
- **Implicit deadlines:** "Let me know your thoughts" without clear urgency
- **Sarcasm/irony:** Tone doesn't match urgency (e.g., "Great, another meeting")
- **Multi-faceted:** Author wants feedback but also informing

Expected score: 0.50–0.70

## Setup & Installation

### Prerequisites
- Python 3.11+
- Docker
- OpenAI API key (or compatible provider via `API_BASE_URL`)

### Local Development

```bash
# Clone repository
git clone https://github.com/Amgothvijaykumar/InboxRL.git
cd Email_agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn pydantic openai httpx

# Run server locally
python -m uvicorn server.app:app --reload --port 8000
```

### Docker Deployment

```bash
# Build image
docker build -t email_triage_agent .

# Run container
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY email_triage_agent
```

## Running Inference

### Set Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4"
export API_BASE_URL="https://api.openai.com/v1"
export ENV_URL="http://localhost:8000"
```

Or for HuggingFace router:

```bash
export OPENAI_API_KEY=$HF_TOKEN
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
```

### Run Baseline

```bash
python inference.py
```

Expected output (JSON logs):
```json
{"type": "START", "task": "email_triage", "env": "email_triage_openenv", "model": "gpt-4"}
{"type": "STEP", "step": 1, "action": "{...}", "reward": 0.75, "done": true}
{"type": "END", "success": true, "steps": 1, "score": 0.75, "total_reward": 0.75, "rewards": [0.75]}
```

## Baseline Performance

**Model:** GPT-4  
**Environment:** 6 tasks across all difficulty levels  
**Average Score:** 0.73  

| Difficulty | Avg Reward | Tasks |
|------------|-----------|-------|
| Easy       | 0.95      | 2     |
| Medium     | 0.72      | 2     |
| Hard       | 0.52      | 2     |
| **Overall**| **0.73**  | **6** |

## API Endpoints

### POST /reset
Reset environment and get initial observation.

**Response:**
```json
{
  "task_id": "easy_spam_001",
  "subject": "FREE MONEY - Click here now!!!",
  "sender": "noreply@spam.com",
  "body": "...",
  "difficulty": "easy"
}
```

### POST /step
Execute action and receive reward.

**Request:**
```json
{
  "label": "junk",
  "draft_reply": null
}
```

**Response:**
```json
{
  "label_score": 1.0,
  "reply_score": 0.0,
  "reward": 0.6,
  "done": true,
  "info": {...}
}
```

### GET /state
Get current environment state (non-hidden info).

**Response:**
```json
{
  "task_id": "easy_spam_001",
  "difficulty": "easy",
  "step_count": 1,
  "done": true
}
```

## Project Structure

```
email_triage_agent/
├── models.py              # Pydantic models (Observation, Action, Reward)
├── server/
│   └── app.py            # FastAPI application
├── inference.py          # Baseline inference script
├── tasks.json            # Email dataset (6 tasks)
├── Dockerfile            # Container configuration
├── openenv.yaml          # OpenEnv specification
└── README.md             # This file
```

## Key Design Decisions

### 1. Hybrid Reward Signal
**60% label accuracy + 40% reply quality**

- Label accuracy: Fast, deterministic, no LLM calls
- Reply quality: Incentivizes thoughtful replies while avoiding false positives

This mirrors real-world RL training where some metrics are cheap (classification accuracy) and others are expensive (human judgment).

### 2. Single-Step Episodes
Each email is one decision. Agent sees email → returns classification + optional reply → receives reward.

This keeps episodes fast (<1 second each) and allows for high-throughput training.

### 3. Difficulty Tiers
Tasks range from trivial (obvious spam) to challenging (ambiguous tone). This ensures:
- Easy wins for policy warmup
- Realistic distribution of email types
- Measurable improvement signal across training

### 4. Multi-Label Support
Agent can attempt reply drafting on **any** email, but only `needs_reply` gets positive reward. This prevents the agent from always replying while allowing it to learn when drafting is useful.

## Troubleshooting

### Docker build fails
Ensure Dockerfile is at project root (not in `server/`).

### Inference script hangs
Check that `ENV_URL` points to running environment:
```bash
curl -X POST http://localhost:8000/reset
```

### Low baseline score
- Verify MODEL_NAME is available on your API provider
- Check OPENAI_API_KEY has correct permissions
- Try a larger model (GPT-4 vs GPT-3.5)

## HuggingFace Spaces Deployment

```bash
# Push to HF Spaces
huggingface-cli repo create email-triage-agent
huggingface-cli upload ./. Amgothvijaykumar/email-triage-agent

# Or use GitHub integration for auto-deploy
# 1. Push to GitHub
# 2. Link repo to HF Space
# 3. Spaces auto-builds on main branch push
```

## Evaluation Criteria

This environment is evaluated on:

1. **Real-world utility (30%):** Email triage is a universal, practical task
2. **Task quality (25%):** 3 difficulty levels with clear rubrics and fair graders
3. **Environment design (20%):** Clean state management, reasonable rewards, proper episodes
4. **Code quality (15%):** OpenEnv compliance, dockerfile works, inference runs
5. **Creativity (10%):** Multi-trajectory potential, interesting failure modes

## Resources

- OpenEnv Docs: https://openenv.dev/
- OpenAI API: https://platform.openai.com/docs/
- HuggingFace Router: https://router.huggingface.co/
- FastAPI: https://fastapi.tiangolo.com/

## License

MIT License — Feel free to build on this environment!

## Author

**InboxRL** — A hackathon submission for building real-world RL environments.

- GitHub: https://github.com/Amgothvijaykumar/InboxRL
- HuggingFace: https://huggingface.co/spaces/Amgothvijaykumar/email-triage-agent
