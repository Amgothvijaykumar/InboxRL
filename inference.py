"""
Email Triage Environment - Baseline Inference Script
Implements proper OpenEnv logging format with [START], [STEP], [END]
Uses OpenAI Client for LLM calls
"""
import asyncio
import json
import os
import sys
from typing import List
import httpx
from openai import OpenAI


# Configuration from environment - REQUIRED
# HF_TOKEN is the primary credential when running on/via HuggingFace
# NO default value for HF_TOKEN (per OpenEnv submission requirements)
HF_TOKEN = os.environ.get("HF_TOKEN")          # no default — must be set explicitly
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_KEY = HF_TOKEN or OPENAI_API_KEY  # HF_TOKEN takes priority

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

# Constants
TASK_NAME = "email_triage"
BENCHMARK = "email_triage_openenv"
MAX_STEPS = 1
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.7


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start in OpenEnv format"""
    print(
        json.dumps({
            "type": "START",
            "task": task,
            "env": env,
            "model": model,
        }),
        flush=True
    )


def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    """Log step execution in OpenEnv format"""
    log_entry = {
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": round(reward, 2),
        "done": done,
    }
    if error:
        log_entry["error"] = error
    print(json.dumps(log_entry), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end in OpenEnv format"""
    print(
        json.dumps({
            "type": "END",
            "success": success,
            "steps": steps,
            "score": round(score, 2),
            "total_reward": round(sum(rewards), 2),
            "rewards": [round(r, 2) for r in rewards],
        }),
        flush=True
    )


async def reset_env(client: httpx.Client) -> dict:
    """Call /reset endpoint"""
    try:
        response = client.post(f"{ENV_URL}/reset", timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[DEBUG] reset() failed: {e}", flush=True)
        raise


async def step_env(client: httpx.Client, action: dict) -> dict:
    """Call /step endpoint"""
    try:
        response = client.post(
            f"{ENV_URL}/step",
            json=action,
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[DEBUG] step() failed: {e}", flush=True)
        raise


def get_model_action(
    client: OpenAI,
    observation: dict, 
    step: int
) -> dict:
    """Use OpenAI Client to generate action from observation"""
    prompt = f"""
You are an email triage agent. Classify this email and optionally draft a reply.

Email:
- Subject: {observation['subject']}
- From: {observation['sender']}
- Body: {observation['body']}

Classify as ONE of: urgent, needs_reply, fyi, junk

If needs_reply is appropriate, draft a brief professional reply (under 100 words).

Respond in this exact JSON format:
{{
  "label": "<urgent|needs_reply|fyi|junk>",
  "draft_reply": "<reply text or null>"
}}
"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=30.0
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Parse JSON response
        action_data = json.loads(text)
        return {
            "label": action_data.get("label", "fyi").lower(),
            "draft_reply": action_data.get("draft_reply"),
        }
    except json.JSONDecodeError:
        print(f"[DEBUG] Failed to parse model response: {text}", flush=True)
        return {"label": "fyi", "draft_reply": None}
    except Exception as e:
        print(f"[DEBUG] OpenAI API request failed: {e}", flush=True)
        return {"label": "fyi", "draft_reply": None}


async def main() -> None:
    """Main inference loop - Multi-episode evaluation"""
    # Verify configuration
    if not API_KEY:
        print("[ERROR] HF_TOKEN or OPENAI_API_KEY environment variable is required", flush=True)
        sys.exit(1)

    # Initialize OpenAI client (works with HF router and OpenAI API)
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )

    http_client = httpx.Client()
    
    # Multi-episode tracking
    # Run at least 1 episode per difficulty level + extra random episodes
    DIFFICULTY_EPISODES = ["easy", "medium", "hard"]  # guaranteed coverage
    EXTRA_EPISODES = 7  # additional random episodes
    all_rewards: List[float] = []
    episode_scores: List[float] = []
    total_steps = 0
    global_step = 0  # sequential step counter for log_step()
    
    print(f"[DEBUG] Using API base: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] Auth via: {'HF_TOKEN' if HF_TOKEN else 'OPENAI_API_KEY'}", flush=True)
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # --- Phase 1: Guaranteed one episode per difficulty level ---
        for difficulty in DIFFICULTY_EPISODES:
            try:
                # Request a specific difficulty via query param (server supports it)
                response = http_client.post(
                    f"{ENV_URL}/reset",
                    params={"difficulty": difficulty},
                    timeout=10.0
                )
                # If server doesn't support ?difficulty param, fall back to plain reset
                if response.status_code == 422:
                    response = http_client.post(f"{ENV_URL}/reset", timeout=10.0)
                response.raise_for_status()
                observation = response.json()

                task_id = observation.get('task_id', 'unknown')
                episode_rewards: List[float] = []

                for step in range(1, MAX_STEPS + 1):
                    global_step += 1
                    action = get_model_action(client, observation, step)
                    result = await step_env(http_client, action)

                    reward = result.get("reward", 0.0)
                    done = result.get("done", False)

                    episode_rewards.append(reward)
                    all_rewards.append(reward)
                    total_steps += 1

                    log_step(step=global_step, action=json.dumps(action), reward=reward, done=done)

                    if done:
                        break

                episode_score = sum(episode_rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
                episode_score = min(max(episode_score, 0.0), 1.0)
                episode_scores.append(episode_score)
                print(f"[DEBUG] Difficulty={difficulty} Task={task_id} Score={episode_score:.2f}", flush=True)

            except Exception as e:
                print(f"[DEBUG] Difficulty={difficulty} episode error: {e}", flush=True)
                episode_scores.append(0.0)

        # --- Phase 2: Extra random episodes ---
        for episode in range(1, EXTRA_EPISODES + 1):
            episode_rewards: List[float] = []
            try:
                observation = await reset_env(http_client)
                task_id = observation.get('task_id', 'unknown')
                difficulty = observation.get('difficulty', 'unknown')

                for step in range(1, MAX_STEPS + 1):
                    global_step += 1
                    action = get_model_action(client, observation, step)
                    result = await step_env(http_client, action)

                    reward = result.get("reward", 0.0)
                    done = result.get("done", False)

                    episode_rewards.append(reward)
                    all_rewards.append(reward)
                    total_steps += 1

                    log_step(step=global_step, action=json.dumps(action), reward=reward, done=done)

                    if done:
                        break

                episode_score = sum(episode_rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
                episode_score = min(max(episode_score, 0.0), 1.0)
                episode_scores.append(episode_score)

                if episode % 3 == 0:
                    print(f"[DEBUG] Extra episode {episode}/{EXTRA_EPISODES}: Score={episode_score:.2f}, Task={task_id}, Difficulty={difficulty}", flush=True)

            except Exception as e:
                print(f"[DEBUG] Extra episode {episode} error: {e}", flush=True)
                episode_scores.append(0.0)
                continue

        # Calculate aggregate score across all episodes
        avg_score = sum(episode_scores) / len(episode_scores) if episode_scores else 0.0
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        success = avg_score >= SUCCESS_SCORE_THRESHOLD
        
        print(f"[DEBUG] === FINAL RESULTS ===", flush=True)
        print(f"[DEBUG] Episodes: {len(episode_scores)}/{num_episodes}", flush=True)
        print(f"[DEBUG] Average Score: {avg_score:.2f}", flush=True)
        print(f"[DEBUG] Average Reward: {avg_reward:.2f}", flush=True)
        print(f"[DEBUG] Total Steps: {total_steps}", flush=True)
        print(f"[DEBUG] Success: {success}", flush=True)

    except Exception as e:
        print(f"[DEBUG] Critical error in main loop: {e}", flush=True)
        avg_score = 0.0
        success = False
        all_rewards = []
        total_steps = 0
        
    finally:
        http_client.close()
        log_end(success=success, steps=total_steps, score=avg_score, rewards=all_rewards)


if __name__ == "__main__":
    asyncio.run(main())
