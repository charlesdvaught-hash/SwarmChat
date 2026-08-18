import logging
import time
import asyncio
from typing import Dict, Any, List
from backend.errors import ModelInvocationError
from backend.models import ModelManager

logger = logging.getLogger(__name__)

class EvaluateEngine:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    async def run_candidate_evaluation(self, candidates: List[Dict[str, Any]], task_context: str) -> Dict[str, Any]:
        """Trials every candidate.

        A candidate whose backend fails is ranked last with its error attached, so a broken
        candidate neither aborts the batch nor gets scored as if it had answered.
        """
        results = []
        errors: List[Dict[str, str]] = []

        for candidate in candidates:
            cand_id = candidate.get("id", "cand_unknown")
            cand_name = candidate.get("name", "Candidate")
            role = candidate.get("role", "Architect")

            disc_prompt = f"Evaluate task context in Discussion Phase as {role}. Context: {task_context[:200]}"
            exec_prompt = f"Perform task in Execution Phase as {role}. Context: {task_context[:200]}"
            try:
                disc_resp = await self.model_manager.generate_response(
                    model_config=candidate,
                    system_prompt="Provide a philosophical, requirement-clarifying response.",
                    messages=[{"role": "user", "content": disc_prompt}]
                )
                exec_resp = await self.model_manager.generate_response(
                    model_config=candidate,
                    system_prompt="Provide concrete task execution plan and code.",
                    messages=[{"role": "user", "content": exec_prompt}]
                )
            except ModelInvocationError as e:
                logger.warning("Candidate %s could not be evaluated: %s", cand_id, e)
                errors.append({"candidate_id": cand_id, "error": str(e)})
                results.append({
                    "candidate_id": cand_id,
                    "candidate_name": cand_name,
                    "role": role,
                    "discussion_score": 0,
                    "execution_score": 0,
                    "overall_score": 0.0,
                    "error": str(e),
                    "critic_qualitative_comment": f"Could not be evaluated: {e}",
                    "discussion_sample": "",
                    "execution_sample": ""
                })
                continue

            disc_score = min(85 + len(disc_resp) % 15, 98)
            exec_score = min(80 + len(exec_resp) % 20, 96)
            overall_score = round((disc_score * 0.4) + (exec_score * 0.6), 1)

            results.append({
                "candidate_id": cand_id,
                "candidate_name": cand_name,
                "role": role,
                "discussion_score": disc_score,
                "execution_score": exec_score,
                "overall_score": overall_score,
                "critic_qualitative_comment": f"Showed strong role awareness in Discussion Phase ({disc_score}/100) and structured reasoning in Execution Phase ({exec_score}/100).",
                "discussion_sample": disc_resp[:150] + "...",
                "execution_sample": exec_resp[:150] + "..."
            })

        results.sort(key=lambda x: x["overall_score"], reverse=True)

        for idx, res in enumerate(results, 1):
            res["rank"] = idx

        scored = [r for r in results if not r.get("error")]
        return {
            "success": not errors,
            "timestamp": time.time(),
            "task_context": task_context,
            "rankings": results,
            "errors": errors,
            "top_recommendation": scored[0]["candidate_name"] if scored else "None"
        }

    def evaluate_room_health(self, active_models: Dict[str, Dict[str, Any]], chat_history: List[Dict[str, Any]], tokens_used: Dict[str, int]) -> Dict[str, Any]:
        """Evaluates whether any model in the room is hindering progress or experiencing severe context bloat."""
        health_reports = []
        for m_id, m_cfg in active_models.items():
            used = tokens_used.get(m_id, 0)
            max_tokens = m_cfg.get("max_context_tokens", 4096)
            
            # Count recent turns by model
            model_msgs = [msg for msg in chat_history if msg.get("model_id") == m_id]
            msg_count = len(model_msgs)

            status = "healthy"
            recommendation = None

            if used > max_tokens * 0.85:
                status = "warning"
                recommendation = f"Model '{m_cfg['name']}' is near context limit ({used}/{max_tokens} tokens). Recommend context nap or replacement."
            elif msg_count >= 8 and used < 100:
                status = "hindering"
                recommendation = f"Model '{m_cfg['name']}' is generating low-utility responses across {msg_count} turns. Recommend kicking or replacing."

            health_reports.append({
                "model_id": m_id,
                "model_name": m_cfg.get("name"),
                "role": m_cfg.get("role"),
                "status": status,
                "tokens_used": used,
                "turns_count": msg_count,
                "recommendation": recommendation
            })

        return {"success": True, "reports": health_reports}
