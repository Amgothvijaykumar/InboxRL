# Email Triage & Smart Reply Agent — OpenM Implementation Roadmap

**Project Goal:** Build a real-world RL environment for email classification, prioritization, and intelligent reply drafting using the OpenM framework.

**Submission Deadline:** April 8, 2026

---

## Phase 1: Project Initialization & Setup

### Objectives
- Create isolated Python environment
- Install OpenM core and dependencies
- Scaffold the project structure via OpenM CLI
- Organize files correctly (critical: Dockerfile at project root)

### Commands
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install openm-core
pip install uv

# Scaffold the environment
openm init email_triage_agent
# Generates: server/, client/, models.py, Dockerfile, inference.py, openm.yaml

# FIX: Move Dockerfile to project root (REQUIRED!)
mv server/Dockerfile ./Dockerfile
```

### Generated Files Structure
```
email_triage_agent/
├── server/
│   └── app.py
├── client/
│   └── client.py
├── models.py              # ← Core logic here
├── Dockerfile             # ← Must be at root
├── inference.py           # ← Update with HF router
├── openm.yaml
├── emails.json            # ← Create dataset
└── venv/
```

### Success Criteria
- [x] Virtual environment activated
- [x] OpenM CLI installed
- [x] Project scaffolded
- [x] Dockerfile location verified at project root

---

## Phase 2: Define Data Models (Pydantic)

### Objectives
- Define Action: what agent outputs (classification + draft reply)
- Define Observation: what agent sees (email content)
- Define State: full environment state (includes hidden ground truth)

### Action Model
```python
from typing import Literal, Optional
from openm.engine.types import Action

class EmailAction(Action):
    """Agent's output: classification + optional draft reply"""
    label: Literal["urgent", "needs_reply", "fyi", "junk"]
    draft_reply: Optional[str] = None
    # Validation: if label == "needs_reply", draft_reply is required
```

### Observation Model
```python
from typing import List
from openm.engine.types import Observation

class EmailObservation(Observation):
    """What the agent sees: email content"""
    subject: str
    sender: str
    body: str
    thread_history: List[str] = []           # Previous messages in thread
    timestamp: str                           # ISO format timestamp
```

### State Model
```python
from openm.engine.types import State

class EmailState(State):
    """Full internal environment state (hidden from agent)"""
    current_email: EmailObservation
    gold_label: str                          # Ground truth label (hidden)
    gold_reply_rubric: dict                  # Evaluation rubric (hidden)
    step_count: int = 0
    done: bool = False
```

### Key Design Notes
- All models extend OpenM base types (Action, Observation, State)
- `gold_*` fields are **never** sent to the agent
- Pydantic validation ensures data integrity
- Observation contains only agent-visible information

### Success Criteria
- [x] All three models defined and inherit from OpenM types
- [x] Type annotations complete
- [x] Validation rules in place

---

## Phase 3: Implement Core Environment Logic

### Objectives
- Implement `reset()` to load fresh episodes
- Implement `step()` to process actions and return rewards
- Implement `get_state()` to expose environment state
- Implement `score_reply()` to judge reply quality

### reset() Function
```python
def reset(self) -> EmailObservation:
    """Load a fresh email episode from dataset"""
    email = random.choice(self.EMAIL_DATASET)
    self.state = EmailState(
        current_email=EmailObservation(**email),
        gold_label=email["gold_label"],
        gold_reply_rubric=email["rubric"],
        step_count=0,
        done=False
    )
    return self.state.current_email
```

**Returns:** EmailObservation (agent sees email, not ground truth)

### step() Function
```python
def step(self, action: EmailAction) -> tuple[EmailObservation, float, bool, dict]:
    """
    Process agent's action and return reward.
    
    Returns:
        observation: Next observation (same email in single-step env)
        reward: Float in [0.0, 1.0]
        done: Episode complete (True after single step)
        info: Additional metadata
    """
    # Score label accuracy (exact match vs ground truth)
    label_score = 1.0 if action.label == self.state.gold_label else 0.0
    
    # Score reply quality (LLM-as-judge with rubric)
    reply_score = 0.0
    if action.draft_reply:
        reply_score = self.score_reply(
            action.draft_reply,
            self.state.gold_reply_rubric
        )
    
    # Combine: 60% label accuracy + 40% reply quality
    reward = 0.6 * label_score + 0.4 * reply_score
    
    self.state.done = True
    self.state.step_count += 1
    
    return self.state.current_email, reward, True, {"label_score": label_score, "reply_score": reply_score}
```

### get_state() Function
```python
def get_state(self) -> EmailState:
    """Return full internal state (for debugging/testing)"""
    return self.state
```

### score_reply() Function
```python
def score_reply(self, draft_reply: str, rubric: dict) -> float:
    """
    Judge reply quality using LLM-as-judge pattern.
    
    Rubric typically includes:
    - tone: "formal", "casual", "friendly"
    - must_include: ["acknowledgment", "action_items", etc]
    - max_words: Max length constraint
    - other domain-specific criteria
    
    Uses small judge model (e.g., Llama-3.1-8B) to score 0.0–1.0
    """
    prompt = f"""
    Score this email reply on a scale of 0.0 to 1.0.
    
    Rubric:
    - Tone: {rubric.get('tone', 'professional')}
    - Must include: {', '.join(rubric.get('must_include', []))}
    - Max words: {rubric.get('max_words', 100)}
    
    Reply:
    {draft_reply}
    
    Return ONLY a float between 0.0 and 1.0.
    """
    
    response = self.judge_client.chat.completions.create(
        model=self.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    score = float(response.choices[0].message.content.strip())
    return max(0.0, min(1.0, score))  # Clamp to [0, 1]
```

### Success Criteria
- [x] reset() loads emails and initializes state
- [x] step() computes correct reward formula
- [x] get_state() exposes environment for inspection
- [x] score_reply() integrates LLM-as-judge evaluator
- [x] All returns are properly typed

---

## Phase 4: Design Reward Signal & Dataset

### Reward Formula
```
reward = 0.6 × label_score + 0.4 × reply_score

Where:
  label_score = 1.0 if action.label == gold_label else 0.0
  reply_score = LLM judge output in [0.0, 1.0] (or 0 if no reply given)
```

**Justification:**
- **60% Label Accuracy:** Binary feedback, deterministic, fast
- **40% Reply Quality:** Learned signal, rewards nuanced performance, uses LLM-as-judge

**Always returns Float in [0.0, 1.0]** ✓

### Dataset Structure (emails.json)
```json
[
  {
    "subject": "URGENT: Server down in prod",
    "sender": "oncall@company.com",
    "body": "Postgres cluster failover occurred at 09:12 UTC. All services returning 503s.",
    "thread_history": [],
    "timestamp": "2025-04-08T09:15:00Z",
    "gold_label": "urgent",
    "rubric": {
      "tone": "formal",
      "must_include": ["acknowledge", "ETA", "next_steps"],
      "max_words": 80
    }
  },
  {
    "subject": "RE: Q2 budget planning",
    "sender": "manager@company.com",
    "body": "Hi, just shared our draft Q2 plan. FYI only — no action needed.",
    "thread_history": ["Original budget doc sent Monday"],
    "timestamp": "2025-04-08T14:30:00Z",
    "gold_label": "fyi",
    "rubric": {
      "tone": "casual",
      "must_include": ["acknowledgment"],
      "max_words": 50
    }
  }
]
```

### Label Distribution (Balanced)
- **urgent** (15%): Production issues, security alerts, high-priority requests
- **needs_reply** (35%): Questions, decisions required, action items
- **fyi** (35%): Announcements, CC'd updates, informational only
- **junk** (15%): Spam, marketing, unrelated noise

### Difficulty Tiers (Multi-Trajectory)
**Easy Emails (Clear intent)**
- Spam indicators: free money, click here, buy now
- Obvious urgent: "DOWN", "CRITICAL", "ASAP"
- Simple FYI: newsletters, status updates

**Medium Emails (Some ambiguity)**
- Thread context required: "RE: discussion" with unclear scope
- Tone indicators: casual vs formal sender patterns
- Mixed urgency: "newsletter but relevant to my project"

**Hard Emails (Complex reasoning)**
- Ambiguous tone: Could be FYI or needs_reply depending on context
- Sarcasm/irony: "Great, another meeting" (negative tone but FYI)
- Reply deadline implicit: "Let me know your thoughts" without explicit timeline
- CC'd to many: Am I stakeholder or bystander?

### Minimum Dataset Size
- **50 emails:** Sufficient for local validation and testing
- **200+ emails:** Full training dataset for real RL runs
- **Ensure balanced distribution** across all 4 labels
- **Include difficulty tiers** for richer learning signal

### Success Criteria
- [x] Dataset contains 50+ emails
- [x] All 4 labels represented with correct percentages
- [x] Rubrics defined for each email
- [x] Thread history populated for relevant emails
- [x] Difficulty tiers mixed throughout

---

## Phase 5: Configure Inference Script & Docker

### Dockerfile (at project root)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir openm-core uv

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Critical:** Must be at project root, NOT in `server/`

### server/app.py — Enable Web UI
```python
from openm import OpenMApp
from models import EmailTriageEnv

app = OpenMApp(
    environment=EmailTriageEnv(),
    enable_web_interface=True  # Enables /web Gradio UI for manual testing
)
```

### inference.py — HuggingFace Router Configuration
```python
import os
import json
import openai
from openm.client import Client

# Configure HuggingFace router (no paid OpenAI key needed!)
client = openai.OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HF_TOKEN")  # Free access token
)

# Specify model available on HuggingFace
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def run_inference():
    """Run inference loop: reset → observe → step → reward"""
    omclient = Client(url="http://localhost:8000")
    
    # Episode loop
    observation = omclient.reset()
    
    # Agent is called here with observation
    # (You'll implement agent logic that uses `client` for LLM inference)
    
    action = generate_action(observation, client, MODEL)
    
    observation, reward, done, info = omclient.step(action)
    
    print(f"Reward: {reward}")
    assert 0.0 <= reward <= 1.0, "Reward must be in [0, 1]"

def generate_action(observation, client, model):
    """Example: Use LLM to generate action from observation"""
    prompt = f"""
    Classify this email:
    
    Subject: {observation.subject}
    From: {observation.sender}
    Body: {observation.body}
    
    Choose ONE: urgent, needs_reply, fyi, junk
    """
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    label = response.choices[0].message.content.strip().lower()
    
    return {
        "label": label,
        "draft_reply": "..." if label == "needs_reply" else None
    }

if __name__ == "__main__":
    run_inference()
```

### Local Test Commands
```bash
# Build Docker image (name must match environment name)
docker build -t email_triage_agent .

# Run container locally
docker run -p 8000:8000 email_triage_agent

# In another terminal, validate structure
openm validate

# Run inference script
uv run inference.py

# Manual testing: Visit http://localhost:8000/web
# - Click reset to load an email
# - Submit an action (label + optional reply)
# - Check returned reward is in [0.0, 1.0]
```

### Success Criteria
- [x] Dockerfile at project root
- [x] Web UI enabled with `enable_web_interface=True`
- [x] inference.py uses HuggingFace router (no paid API key)
- [x] Inference loop completes without errors
- [x] All test commands execute successfully

---

## Phase 6: Local Testing & HuggingFace Deployment

### Local Testing Checklist
```bash
# 1. Validate file structure
openm validate
# Output should confirm:
# - models.py exists
# - server/app.py exists
# - Dockerfile at project root
# - inference.py exists

# 2. Build and run Docker
docker build -t email_triage_agent .
docker run -p 8000:8000 email_triage_agent

# 3. Run inference end-to-end
uv run inference.py
# Should see: Reward: 0.XX (float between 0 and 1)

# 4. Manual testing via Web UI
# Open browser: http://localhost:8000/web
# - Click "Reset" to load an email
# - Observe email subject, sender, body
# - Submit action (classification + optional reply)
# - Verify reward returned is in [0.0, 1.0]
# - Test multiple emails across difficulty tiers
```

### Deployment to HuggingFace Spaces
```bash
# 1. Generate access token
# Go to: https://huggingface.co/settings/tokens
# Create new token (read + write permissions)

# 2. Set environment variable
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx

# 3. Deploy with one command
openm push --username YOUR_HF_USERNAME --name email_triage_agent

# 4. Get your submission URL
# After deploy, Space URL: https://huggingface.co/spaces/YOUR_USERNAME/email_triage_agent
```

### What Gets Deployed
- **Docker Image** → HuggingFace Space (runs your environment)
- **REST API** → `/reset`, `/step`, `/state` endpoints
- **Gradio UI** → `/web` interactive interface
- **Public URL** → Share for submission

### Pre-Submission Checklist
- [ ] `openm validate` passes
- [ ] `uv run inference.py` completes successfully
- [ ] Web UI at `/web` functions correctly:
  - [ ] Reset loads email
  - [ ] Step processes action
  - [ ] Reward returned in [0.0, 1.0]
- [ ] Docker builds and runs locally
- [ ] HuggingFace Space deployed and publicly accessible
- [ ] All 4 labels present in dataset
- [ ] Difficulty tiers represented
- [ ] Rubrics defined for all emails

### Success Criteria
- [x] Environment passes local validation
- [x] Docker image builds without errors
- [x] Inference loop completes end-to-end
- [x] Web UI interactive and responsive
- [x] HuggingFace Spaces deployment successful
- [x] Public URL ready for submission

---

## Key Design Decisions

### 1. Hybrid Reward Signal
The **0.6 × label + 0.4 × reply** formula is deliberately balanced:
- Label accuracy (60%): Fast, deterministic, no model calls
- Reply quality (40%): Uses LLM-as-judge, allows learning beyond classification

This mimics real-world RL post-training where you have verifiable metrics + human-in-loop feedback.

### 2. Dataset Complexity
Your 50-email dataset is the **biggest leverage point**:
- Easy tier: Spam filters, obvious alerts (agent learns baseline)
- Medium tier: Mixed signals, context-dependent (agent learns trade-offs)
- Hard tier: Ambiguous cases, implicit deadlines (agent learns nuance)

More diverse data = richer training signal for post-training.

### 3. Dockerfile Location (Critical!)
**Must be at project root.** This is the most common failure point. After scaffolding:
```bash
mv server/Dockerfile ./Dockerfile
```
Verify with: `ls -la Dockerfile` before any Docker build.

### 4. LLM-as-Judge Pattern
The `score_reply()` function uses a small judge model to score replies 0.0–1.0:
- Rubric-guided evaluation (tone, required content, length)
- No training needed — responds to structured prompts
- Can use free HuggingFace router

This is exactly what enables RL post-training for reply generation.

### 5. Single-Step Episodes
Each email is one step: reset → observe → act → receive reward → done.
This simplifies the environment but captures the core task. Multi-trajectory could be added later (e.g., agent gets feedback and revises).

---

## Common Pitfalls & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Docker build fails | Dockerfile in `server/` subdirectory | Move to project root: `mv server/Dockerfile ./` |
| `openm validate` fails | Missing files (models.py, app.py, etc) | Check all files exist in correct locations |
| Inference loop hangs | HF_TOKEN not set or invalid | `export HF_TOKEN=hf_xxx` before running |
| Web UI unresponsive | `enable_web_interface=False` | Set to `True` in server/app.py |
| Reward outside [0, 1] | `score_reply()` returns invalid value | Clamp: `max(0.0, min(1.0, score))` |
| HuggingFace push fails | Wrong HF username or token | Verify: `openm push --help` and token permissions |

---

## Timeline

**Day 1:** Phases 1–2 (Setup + Design)
**Day 2:** Phase 3–4 (Implementation + Dataset)
**Day 3:** Phase 5–6 (Testing + Deployment)

**Total: 3 days to submission.**

---

## Resources

- **OpenM Docs:** https://openim.openms.org/
- **HuggingFace Router:** https://router.huggingface.co/
- **Pydantic Docs:** https://docs.pydantic.dev/
- **Uvicorn Docs:** https://www.uvicorn.org/

---

**Next Step:** Execute Phase 1 — Project Initialization & Setup
