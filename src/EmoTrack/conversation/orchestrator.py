from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
import json

from tqdm import tqdm

from EmoTrack.client.evermem_agent import EverMemClientAgent
from EmoTrack.conversation.config import ConversationConfig
from EmoTrack.conversation.model import RetryableGenerationError
from EmoTrack.counselor.evermem_agent import EverMemCounselorAgent
from EmoTrack.evermemos.client import EverMemOSClient, MemorySearchResult
from EmoTrack.latent_variable.trend import ResolvedLatentState
from EmoTrack.utils.dialogue_text import extract_vertex_history_parts


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConversationSessionSummary:
    session_id: int
    turn_count: int
    summary: str
    key_facts: list[str]
    goals: list[str]
    homework: list[str]
    risk_flags: list[str]


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


SELF_REPORT_SCORES_NAME = "self_report.json"
_EXPLICIT_LONGITUDINAL_REFERENCE_RE = re.compile(
    r"\b("
    r"last time|last session|previous session|previous sessions|"
    r"earlier session|earlier sessions|we talked about|"
    r"you mentioned before|as we discussed|back then"
    r")\b|上次|之前提到|之前说到|前一次|前几次",
    flags=re.IGNORECASE,
)


class ConversationOrchestrator:
    def __init__(
        self,
        *,
        config: ConversationConfig,
        run_dir: Path,
        cache_dir: Path,
        evermem_env_path: Path,
        group_id: str,
        client_profile: dict[str, Any],
        counselor_profile: dict[str, Any],
        client_agent: EverMemClientAgent,
        counselor_agent: EverMemCounselorAgent,
        evermem_client: EverMemOSClient,
        model_metadata: dict[str, Any],
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.cache_dir = cache_dir
        self.evermemos_env_path = evermem_env_path
        self.group_id = group_id
        self.client_profile = client_profile
        self.counselor_profile = counselor_profile
        self.client_agent = client_agent
        self.counselor_agent = counselor_agent
        self.evermem_client = evermem_client
        self.model_metadata = model_metadata

        common_dir = self.run_dir / "common"
        common_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_writer = JsonlWriter(common_dir / "transcript.jsonl")
        self.thinking_writer = JsonlWriter(self.run_dir / "thinking.jsonl")
        self.memory_writer = JsonlWriter(self.run_dir / "memory.jsonl")
        self.failure_writer = JsonlWriter(self.run_dir / "failure.jsonl")
        self.rejected_output_writer = JsonlWriter(self.run_dir / "rejected_outputs.jsonl")
        self.progress_path = self.run_dir / "progress.json"
        self.meta_path = self.run_dir / "meta.json"
        self.eval_ready_path = self.run_dir / "eval_ready_case.json"
        self.summary_md_path = self.run_dir / "summary.md"
        self.session_latent_plan_path = self.run_dir / "session_latent_plan.json"
        self.self_report_scores_path = self.run_dir / "common" / SELF_REPORT_SCORES_NAME
        self.steps_dir = self.cache_dir / "steps"
        self._attach_invalid_output_loggers()

    def _log_progress(self, message: str) -> None:
        print(f"[conversation] run_id={self.run_dir.name} {message}", flush=True)

    def _attach_invalid_output_loggers(self) -> None:
        for agent in (self.client_agent, self.counselor_agent):
            setter = getattr(agent, "set_invalid_output_logger", None)
            if callable(setter):
                setter(self._record_rejected_output)

    def _record_rejected_output(self, payload: dict[str, Any]) -> None:
        normalized = dict(payload)
        normalized.setdefault("run_id", self.run_dir.name)
        normalized.setdefault("timestamp", now_iso())
        self.rejected_output_writer.write(normalized)

    def _session_counts(self, progress: dict[str, Any]) -> tuple[int, int]:
        completed = int(progress.get("completed_session_count", 0) or 0)
        total = int(progress.get("num_sessions", self.config.simulation.num_sessions) or 0)
        return completed, total

    def _group_name(self) -> str:
        return f"EmoTrack Longitudinal Conversation [{self.run_dir.name}]"

    def _session_progress_label(self, session_id: int) -> str:
        return f"{self.run_dir.name}:s{session_id:03d}"

    def _initial_progress(self) -> dict[str, Any]:
        return {
            "run_id": self.run_dir.name,
            "group_id": self.group_id,
            "started_at": now_iso(),
            "completed_at": None,
            "status": "running",
            "num_sessions": self.config.simulation.num_sessions,
            "completed_session_count": 0,
            "session_summaries": [],
            "eval_sessions": [],
            "transcript": [],
            "steps": [],
            "current_session_close_pending": False,
            "current_session_end_reason": "",
        }

    def _load_progress(self) -> dict[str, Any]:
        if not self.progress_path.exists() or not self.config.run.resume:
            return self._initial_progress()

        raw = json.loads(self.progress_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Progress file must contain a JSON object: {self.progress_path}")
        progress_num_sessions = int(raw.get("num_sessions", 0))
        config_num_sessions = self.config.simulation.num_sessions
        if progress_num_sessions != config_num_sessions:
            raise ValueError(
                "Progress/config mismatch for simulation.num_sessions: "
                f"progress={progress_num_sessions}, config={config_num_sessions}. "
                "Use a new run_id/output directory or disable resume."
            )
        return self._normalize_loaded_progress(raw)

    def _normalize_loaded_progress(self, progress: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(progress)
        normalized.pop("current_complaint", None)
        normalized.pop("baseline_complaint", None)

        transcript = normalized.get("transcript")
        if isinstance(transcript, list):
            normalized["transcript"] = [
                {key: value for key, value in row.items() if key != "complaint"}
                if isinstance(row, dict)
                else row
                for row in transcript
            ]

        session_summaries = normalized.get("session_summaries")
        if isinstance(session_summaries, list):
            normalized["session_summaries"] = [
                {key: value for key, value in item.items() if key != "complaint"}
                if isinstance(item, dict)
                else item
                for item in session_summaries
            ]

        summaries = normalized.get("session_summaries", [])
        summary_count = len(summaries) if isinstance(summaries, list) else 0
        completed = max(int(normalized.get("completed_session_count", 0) or 0), summary_count)
        completed = min(completed, self.config.simulation.num_sessions)
        normalized["completed_session_count"] = completed

        if completed >= self.config.simulation.num_sessions:
            normalized["status"] = "completed"
        else:
            normalized["status"] = "running"
            normalized["completed_at"] = None
        return normalized

    def _save_progress(self, payload: dict[str, Any]) -> None:
        self.progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _serialize_summary(self, summary: ConversationSessionSummary) -> dict[str, Any]:
        return asdict(summary)

    def _parse_summary(self, raw: dict[str, Any]) -> ConversationSessionSummary:
        return ConversationSessionSummary(
            session_id=int(raw["session_id"]),
            turn_count=int(raw["turn_count"]),
            summary=str(raw["summary"]).strip(),
            key_facts=[str(item) for item in raw.get("key_facts", [])],
            goals=[str(item) for item in raw.get("goals", [])],
            homework=[str(item) for item in raw.get("homework", [])],
            risk_flags=[str(item) for item in raw.get("risk_flags", [])],
        )

    def _session_rows(self, progress: dict[str, Any], session_id: int) -> list[dict[str, Any]]:
        rows = progress.get("transcript", [])
        if not isinstance(rows, list):
            raise TypeError("Progress transcript must be a list")
        return [row for row in rows if int(row.get("session_id", 0)) == session_id]

    def _session_history(self, progress: dict[str, Any], session_id: int) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for row in self._session_rows(progress, session_id):
            item: dict[str, Any] = {"speaker": str(row["speaker"]), "text": str(row["text"])}
            parts = row.get("vertex_history_parts")
            if isinstance(parts, list) and parts:
                item["vertex_history_parts"] = parts
            history.append(item)
        return history

    @staticmethod
    def _clip_text(text: str, *, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return ""
        if len(normalized) <= max_chars:
            return normalized
        clipped = normalized[: max_chars - 3].rstrip()
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        return clipped.rstrip(",;:") + "..."

    @staticmethod
    def _format_summary_note(summary: ConversationSessionSummary) -> str:
        lines = [ConversationOrchestrator._clip_text(summary.summary, max_chars=280)]
        carry_forward = [
            ConversationOrchestrator._clip_text(item, max_chars=90)
            for item in summary.key_facts
            if str(item).strip()
        ][:2]
        if carry_forward:
            lines.append("Carry forward: " + "; ".join(carry_forward))
        return "\n".join(line for line in lines if line.strip()).strip()

    def _latest_summary_note(self, progress: dict[str, Any]) -> str:
        summaries = progress.get("session_summaries", [])
        if not isinstance(summaries, list) or not summaries:
            return ""
        return self._format_summary_note(self._parse_summary(summaries[-1]))

    def _build_carryover_note(self, progress: dict[str, Any]) -> str:
        return self._latest_summary_note(progress)

    @staticmethod
    def _longitudinal_memories(
        *,
        session_history: list[dict[str, str]],
        retrieved_memories: list[str],
    ) -> list[str]:
        if not retrieved_memories:
            return []
        in_session_texts = {
            str(item.get("text", "")).strip()
            for item in session_history
            if str(item.get("text", "")).strip()
        }
        return [snippet for snippet in retrieved_memories if snippet.strip() and snippet.strip() not in in_session_texts]

    @staticmethod
    def _extract_numeric_scores(raw: Any) -> list[float]:
        scores: list[float] = []

        def visit(value: Any) -> None:
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, (int, float)):
                scores.append(float(value))
                return
            if isinstance(value, dict):
                for child in value.values():
                    visit(child)
                return
            if isinstance(value, list):
                for child in value:
                    visit(child)

        visit(raw)
        return scores

    @staticmethod
    def _has_explicit_longitudinal_reference(query: str) -> bool:
        normalized = str(query or "").strip()
        if not normalized:
            return False
        return _EXPLICIT_LONGITUDINAL_REFERENCE_RE.search(normalized) is not None

    def _strong_retrieval_score_threshold(self) -> float:
        return max(float(self.config.simulation.retrieval_threshold) + 0.10, 0.25)

    def _max_retrieval_score(self, search_result: MemorySearchResult) -> float | None:
        raw_result = search_result.raw_result if isinstance(search_result.raw_result, dict) else {}
        result = raw_result.get("result", {}) if isinstance(raw_result.get("result"), dict) else {}
        scores = self._extract_numeric_scores(result.get("scores", []))
        if not scores:
            return None
        return max(scores)

    def _prompt_retrieved_memories(
        self,
        *,
        turn_id: int,
        query: str,
        session_history: list[dict[str, str]],
        search_result: MemorySearchResult,
    ) -> list[str]:
        prompt_candidates = search_result.prompt_snippets or [
            snippet
            for snippet in search_result.snippets
            if self._is_prompt_safe_memory(snippet)
        ]
        longitudinal_memories = self._longitudinal_memories(
            session_history=session_history,
            retrieved_memories=prompt_candidates,
        )
        if not longitudinal_memories:
            return []
        if turn_id == 1:
            return longitudinal_memories
        if self._has_explicit_longitudinal_reference(query):
            return longitudinal_memories
        max_score = self._max_retrieval_score(search_result)
        if max_score is not None and max_score >= self._strong_retrieval_score_threshold():
            return longitudinal_memories
        return []

    @staticmethod
    def _is_prompt_safe_memory(text: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return False
        if "\n" in str(text or ""):
            return False
        return len(normalized.split()) <= 24 and len(normalized) <= 180

    def _summary_generation_max_tokens(self) -> int:
        return 1024

    def _counselor_generation_max_tokens(self) -> int:
        value = int(self.config.simulation.counselor_max_tokens)
        if value <= 0:
            raise ValueError("simulation.counselor_max_tokens must be > 0")
        return value

    def _client_generation_max_tokens(self) -> int:
        value = int(self.config.simulation.client_max_tokens)
        if value <= 0:
            raise ValueError("simulation.client_max_tokens must be > 0")
        return value

    def _current_session_id(self, progress: dict[str, Any]) -> int:
        return int(progress.get("completed_session_count", 0)) + 1

    def _next_step(self, progress: dict[str, Any], session_id: int) -> tuple[str, int]:
        rows = self._session_rows(progress, session_id)
        if not rows:
            return "Counselor", 1
        last = rows[-1]
        last_speaker = str(last["speaker"])
        last_turn_id = int(last["turn_id"])
        if last_speaker == "Counselor":
            return "Client", last_turn_id
        return "Counselor", last_turn_id + 1

    def _write_step_cache(self, payload: dict[str, Any]) -> Path:
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        step_number = len(list(self.steps_dir.glob("*.json"))) + 1
        path = self.steps_dir / f"{step_number:04d}_{payload['speaker'].lower()}_s{payload['session_id']:03d}_t{payload['turn_id']:03d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _append_transcript(self, row: dict[str, Any]) -> None:
        self.transcript_writer.write(row)

    def _append_thinking(self, row: dict[str, Any]) -> None:
        self.thinking_writer.write(row)

    def _append_memory_event(self, row: dict[str, Any]) -> None:
        self.memory_writer.write(row)

    def _save_meta(self, progress: dict[str, Any]) -> None:
        payload = {
            "run_id": self.run_dir.name,
            "group_id": self.group_id,
            "status": progress["status"],
            "started_at": progress["started_at"],
            "completed_at": progress.get("completed_at"),
            "config": self.config.to_dict(),
            "models": self.model_metadata,
            "evermemos_env_path": str(self.evermemos_env_path),
            "completed_session_count": progress["completed_session_count"],
            "cache_dir": str(self.cache_dir),
            "session_latent_plan": str(self.session_latent_plan_path),
        }
        self.meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_str(mapping: dict[str, Any], key: str, default: str = "") -> str:
        value = mapping.get(key, default)
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _read_list(mapping: dict[str, Any], key: str) -> list[Any]:
        value = mapping.get(key, [])
        return value if isinstance(value, list) else []

    def _build_client_info(self) -> dict[str, Any]:
        client_info = self.client_profile.get("client_info", {})
        if isinstance(client_info, dict) and client_info:
            return client_info

        intake_profile = self.client_profile.get("intake_profile", {})
        if isinstance(intake_profile, dict) and intake_profile:
            return intake_profile

        portrait = self.client_profile.get("portrait", {})
        report = self.client_profile.get("report", {})
        if not isinstance(portrait, dict):
            portrait = {}
        if not isinstance(report, dict):
            report = {}

        marital_status = self._read_str(portrait, "marital_status")
        if not marital_status:
            marital_status = self._read_str(portrait, "martial_status")

        static_traits = {
            "age": self._read_str(portrait, "age"),
            "name": self._read_str(portrait, "name"),
            "gender": self._read_str(portrait, "gender"),
            "occupation": self._read_str(portrait, "occupation"),
            "educational_background": self._read_str(portrait, "educational_background"),
            "marital_status": marital_status,
            "family_status": self._read_str(portrait, "family_status"),
            "social_status": self._read_str(portrait, "social_status"),
            "medical_history": self._read_str(portrait, "medical_history"),
            "language_features": self._read_str(portrait, "language_features"),
        }

        main_problem = self._read_str(report, "main_problem") or self._read_str(portrait, "symptoms")
        topic = self._read_str(report, "topic") or self._read_str(portrait, "primary_topic")
        return {
            "static_traits": static_traits,
            "main_problem": main_problem,
            "topic": topic,
            "core_demands": self._read_str(report, "core_demands"),
            "growth_experiences": self._read_list(report, "growth_experiences"),
            "core_beliefs": self._read_list(report, "core_beliefs"),
            "special_situations": self._read_list(report, "special_situations"),
        }

    def _save_eval_ready_case(self, progress: dict[str, Any]) -> None:
        payload = {
            "client_id": self.run_dir.name,
            "client_info": self._build_client_info(),
            "sessions": progress.get("eval_sessions", []),
        }
        self.eval_ready_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _save_summary_markdown(self, progress: dict[str, Any]) -> None:
        summaries = [self._parse_summary(item) for item in progress.get("session_summaries", [])]
        lines = [f"# Run Summary ({self.run_dir.name})", ""]
        for summary in summaries:
            lines.append(f"## Session {summary.session_id}")
            lines.append(f"- Turns: {summary.turn_count}")
            lines.append(f"- Summary: {summary.summary}")
            lines.append(f"- Key facts: {', '.join(summary.key_facts) if summary.key_facts else 'None'}")
            lines.append(f"- Goals: {', '.join(summary.goals) if summary.goals else 'None'}")
            lines.append(f"- Homework: {', '.join(summary.homework) if summary.homework else 'None'}")
            lines.append(f"- Risk flags: {', '.join(summary.risk_flags) if summary.risk_flags else 'None'}")
            lines.append("")
        self.summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    def _save_outputs(self, progress: dict[str, Any]) -> None:
        self._save_progress(progress)
        self._save_meta(progress)
        self._save_eval_ready_case(progress)
        self._save_summary_markdown(progress)
        self._save_session_latent_plan()

    def _load_self_report_sessions(self) -> list[dict[str, Any]]:
        if not self.self_report_scores_path.exists():
            return []
        try:
            payload = json.loads(self.self_report_scores_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
        return sessions if isinstance(sessions, list) else []

    @staticmethod
    def _self_report_score_complete(score: Any) -> bool:
        if not isinstance(score, dict):
            return False
        samples = score.get("samples")
        try:
            sample_count = int(score.get("sample_count") or 0)
        except (TypeError, ValueError):
            sample_count = 0
        return (
            sample_count > 0
            and isinstance(samples, list)
            and bool(samples)
            and isinstance(score.get("symptom_scores"), dict)
            and "overall_score" in score
        )

    def _self_report_sessions_by_id(self) -> dict[int, dict[str, Any]]:
        by_session: dict[int, dict[str, Any]] = {}
        for item in self._load_self_report_sessions():
            if not isinstance(item, dict):
                continue
            try:
                session_id = int(item.get("session_id", 0) or 0)
            except (TypeError, ValueError):
                continue
            if session_id > 0:
                by_session[session_id] = item
        return by_session

    def _incomplete_self_report_session_ids(self, progress: dict[str, Any]) -> list[int]:
        completed = min(
            int(progress.get("completed_session_count", 0) or 0),
            self.config.simulation.num_sessions,
        )
        if completed <= 0:
            return []
        by_session = self._self_report_sessions_by_id()
        incomplete: list[int] = []
        for session_id in range(1, completed + 1):
            raw_session = by_session.get(session_id, {})
            if not self._self_report_score_complete(raw_session.get("self_report_score")):
                incomplete.append(session_id)
        return incomplete

    def _summary_note_before_session(self, progress: dict[str, Any], session_id: int) -> str:
        summaries: list[ConversationSessionSummary] = []
        for item in progress.get("session_summaries", []):
            if not isinstance(item, dict):
                continue
            try:
                summary = self._parse_summary(item)
            except Exception:
                continue
            if summary.session_id < session_id:
                summaries.append(summary)
        if not summaries:
            return ""
        latest = max(summaries, key=lambda item: item.session_id)
        return self._format_summary_note(latest)

    def _repair_incomplete_self_reports(self, progress: dict[str, Any]) -> list[int]:
        repaired: list[int] = []
        for session_id in self._incomplete_self_report_session_ids(progress):
            events = self._session_history(progress, session_id)
            if not events:
                raise ValueError(f"Cannot repair self-report for empty session {session_id}")
            try:
                self_report_score = self.client_agent.self_report_session(
                    session_id=session_id,
                    session_history=events,
                    short_term_note=self._summary_note_before_session(progress, session_id),
                    max_tokens=max(1024, min(1536, self._client_generation_max_tokens())),
                    resolved_latent_state=self.client_agent.resolve_latent_state(session_id),
                )
            except RetryableGenerationError:
                raise
            except Exception as exc:
                raise RetryableGenerationError(
                    f"Self-report generation failed for session {session_id}: {exc}"
                ) from exc
            self._save_self_report_score_entry(session_id=session_id, self_report_score=self_report_score)
            repaired.append(session_id)
        return repaired

    def _save_self_report_score_entry(
        self,
        *,
        session_id: int,
        self_report_score: dict[str, Any],
    ) -> None:
        sessions = [
            item
            for item in self._load_self_report_sessions()
            if int(item.get("session_id", 0) or 0) != session_id
        ]
        sessions.append(
            {
                "session_id": session_id,
                "session_index": session_id,
                "captured_at": now_iso(),
                "self_report_score": self_report_score,
            }
        )
        sessions.sort(key=lambda item: int(item.get("session_id", 0) or 0))
        payload = {
            "run_id": self.run_dir.name,
            "scale": "PHQ-8",
            "sessions": sessions,
        }
        self.self_report_scores_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _resolved_latent_sessions(self) -> list[ResolvedLatentState]:
        return [self.client_agent.resolve_latent_state(session_id) for session_id in range(1, self.config.simulation.num_sessions + 1)]

    def _save_session_latent_plan(self) -> None:
        resolved_sessions = self._resolved_latent_sessions()
        payload = {
            "run_id": self.run_dir.name,
            "trend_plan": {
                "version": self.client_agent.trend_plan.version,
                "description": self.client_agent.trend_plan.description,
                "targets": {
                    target.path: (
                        {
                            "name": target.trend.name,
                            "values": list(target.trend.schedule),
                            "abbreviation": target.trend.abbreviation,
                        }
                        if target.trend.schedule
                        else {
                            "name": target.trend.name,
                            "scale": target.trend.scale,
                            "period": target.trend.period,
                            "phase": target.trend.phase,
                            "abbreviation": target.trend.abbreviation,
                        }
                    )
                    for target in self.client_agent.trend_plan.targets
                },
            },
            "sessions": [session.to_dict() for session in resolved_sessions],
        }
        self.session_latent_plan_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _message_id(self, session_id: int, turn_id: int, speaker: str) -> str:
        return f"{self.run_dir.name}_s{session_id:03d}_t{turn_id:03d}_{speaker.lower()}"

    @staticmethod
    def _serialize_latent_snapshot(
        resolved_latent_state: ResolvedLatentState | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if resolved_latent_state is None:
            return {}, []
        return (
            resolved_latent_state.hidden_state,
            [item.to_dict() for item in resolved_latent_state.applied_trends],
        )

    def _record_step(
        self,
        *,
        progress: dict[str, Any],
        session_id: int,
        turn_id: int,
        speaker: str,
        text: str,
        thinking: str,
        query: str,
        prompt_messages: list[dict[str, str]],
        search_result: MemorySearchResult,
        prompt_retrieved_memories: list[str],
        store_response: dict[str, Any],
        raw_message: dict[str, Any] | None = None,
        resolved_latent_state: ResolvedLatentState | None = None,
    ) -> None:
        message_id = self._message_id(session_id, turn_id, speaker)
        session_hidden_state, session_trends = self._serialize_latent_snapshot(resolved_latent_state)
        transcript_row = {
            "session_id": session_id,
            "session_index": session_id,
            "turn_id": turn_id,
            "turn_index": turn_id,
            "speaker": speaker,
            "text": text,
            "message_id": message_id,
            "session_hidden_state": session_hidden_state,
            "session_trends": session_trends,
        }
        vertex_history_parts = extract_vertex_history_parts(raw_message)
        if vertex_history_parts:
            transcript_row["vertex_history_parts"] = vertex_history_parts
        progress["transcript"].append(transcript_row)
        progress["steps"].append(
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "message_id": message_id,
            }
        )
        self._append_transcript(transcript_row)
        self._append_thinking(
            {
                "session_id": session_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "message_id": message_id,
                "thinking": thinking,
            }
        )
        self._append_memory_event(
            {
                "event_type": "search_and_store",
                "session_id": session_id,
                "turn_id": turn_id,
                "speaker": speaker,
                "query": query,
                "retrieved_memories": search_result.snippets,
                "prompt_retrieved_memories": prompt_retrieved_memories,
                "message_id": message_id,
                "store_response": store_response,
                "timestamp": now_iso(),
            }
        )
        step_payload = {
            "session_id": session_id,
            "turn_id": turn_id,
            "speaker": speaker,
            "message_id": message_id,
            "query": query,
            "prompt_messages": prompt_messages,
            "retrieved_memories": search_result.snippets,
            "prompt_retrieved_memories": prompt_retrieved_memories,
            "evermemos_search_result": search_result.raw_result,
            "visible_text": text,
            "thinking": thinking,
            "raw_message": raw_message or {},
            "evermemos_store_response": store_response,
            "session_hidden_state": session_hidden_state,
            "session_trends": session_trends,
        }
        cache_path = self._write_step_cache(step_payload)
        progress["steps"][-1]["cache_path"] = str(cache_path)

    def _store_message(
        self,
        *,
        session_id: int,
        turn_id: int,
        speaker: str,
        text: str,
        refer_list: list[str],
    ) -> dict[str, Any]:
        return self.evermem_client.store_message(
            message_id=self._message_id(session_id, turn_id, speaker),
            create_time=now_iso(),
            sender=speaker,
            sender_name=speaker,
            role="assistant" if speaker == "Counselor" else "user",
            content=text,
            group_id=self.group_id,
            group_name=self._group_name(),
            scene="assistant",
            refer_list=refer_list,
        )

    def _search(self, *, query: str) -> MemorySearchResult:
        return self.evermem_client.search_memories(
            query=query,
            group_id=self.group_id,
            top_k=self.config.simulation.retrieval_top_k,
            retrieve_method=self.config.simulation.retrieval_method,
            memory_types=["event_log", "episodic_memory"],
            score_threshold=self.config.simulation.retrieval_threshold,
        )

    def _finalize_session(
        self,
        progress: dict[str, Any],
        session_id: int,
        resolved_latent_state: ResolvedLatentState | None,
    ) -> None:
        events = self._session_history(progress, session_id)
        if not events:
            raise ValueError(f"Cannot finalize empty session {session_id}")
        prior_short_term_note = self._latest_summary_note(progress)
        try:
            summary_raw = self.counselor_agent.summarize_session(
                session_id=session_id,
                events=events,
                prior_short_term_note=prior_short_term_note,
                max_tokens=self._summary_generation_max_tokens(),
            )
        except RetryableGenerationError:
            raise
        except Exception as exc:
            raise RetryableGenerationError(
                f"Session summary generation failed for session {session_id}: {exc}"
            ) from exc
        summary = ConversationSessionSummary(
            session_id=session_id,
            turn_count=len([item for item in events if item["speaker"] == "Counselor"]),
            summary=str(summary_raw["summary"]).strip(),
            key_facts=[str(item) for item in summary_raw["key_facts"]],
            goals=[str(item) for item in summary_raw["goals"]],
            homework=[str(item) for item in summary_raw["homework"]],
            risk_flags=[str(item) for item in summary_raw["risk_flags"]],
        )

        try:
            self_report_score = self.client_agent.self_report_session(
                session_id=session_id,
                session_history=events,
                short_term_note=prior_short_term_note,
                max_tokens=max(1024, min(1536, self._client_generation_max_tokens())),
                resolved_latent_state=resolved_latent_state,
            )
        except RetryableGenerationError:
            raise
        except Exception as exc:
            raise RetryableGenerationError(
                f"Self-report generation failed for session {session_id}: {exc}"
            ) from exc
        self._save_self_report_score_entry(session_id=session_id, self_report_score=self_report_score)

        dialogue = []
        for row in self._session_rows(progress, session_id):
            role = "Counselor" if row["speaker"] == "Counselor" else "Client"
            dialogue.append({"role": role, "text": row["text"]})
        progress["session_summaries"] = [
            item
            for item in progress.get("session_summaries", [])
            if int(item.get("session_id", 0) or 0) != session_id
        ]
        progress["session_summaries"].append(self._serialize_summary(summary))
        progress["session_summaries"].sort(key=lambda item: int(item.get("session_id", 0) or 0))
        progress["eval_sessions"] = [
            item
            for item in progress.get("eval_sessions", [])
            if int(item.get("session_number", 0) or 0) != session_id
        ]
        progress["eval_sessions"].append({"session_number": session_id, "session_dialogue": dialogue})
        progress["eval_sessions"].sort(key=lambda item: int(item.get("session_number", 0) or 0))
        progress["completed_session_count"] = len(progress["session_summaries"])
        progress["current_session_close_pending"] = False
        progress["current_session_end_reason"] = ""

        self._append_memory_event(
            {
                "event_type": "session_end",
                "session_id": session_id,
                "summary": summary.summary,
                "timestamp": now_iso(),
            }
        )

    def _ensure_conversation_meta(self, progress: dict[str, Any]) -> None:
        self.evermem_client.ensure_conversation_meta(
            group_id=self.group_id,
            name=self._group_name(),
            scene="assistant",
            created_at=progress["started_at"],
            default_timezone="UTC",
            user_details={
                "Client": {"full_name": "Client", "role": "user", "extra": {}},
                "Counselor": {"full_name": "Counselor", "role": "assistant", "extra": {}},
            },
            description="EmoTrack longitudinal conversation",
            tags=["emotrack", "evermemos", "longitudinal"],
        )

    def run(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        resuming = self.progress_path.exists() and self.config.run.resume
        progress = self._load_progress()
        if resuming and self._current_session_id(progress) <= self.config.simulation.num_sessions:
            self._save_outputs(progress)
        self._ensure_conversation_meta(progress)
        self._save_session_latent_plan()
        completed_sessions, total_sessions = self._session_counts(progress)
        self._log_progress(
            f"{'resume' if resuming else 'start'} status={progress.get('status', 'running')} "
            f"sessions={completed_sessions}/{total_sessions}"
        )

        session_pbar: tqdm | None = None
        try:
            repaired_self_reports = self._repair_incomplete_self_reports(progress)
            if repaired_self_reports:
                self._save_outputs(progress)
                self._log_progress(
                    "repaired self-report sessions="
                    + ",".join(str(session_id) for session_id in repaired_self_reports)
                )
            announced_session_id = 0
            while self._current_session_id(progress) <= self.config.simulation.num_sessions:
                session_id = self._current_session_id(progress)
                if session_id != announced_session_id:
                    if session_pbar is not None:
                        session_pbar.close()
                        session_pbar = None
                    announced_session_id = session_id
                    self._log_progress(
                        f"session {session_id}/{self.config.simulation.num_sessions} running"
                    )
                    history = self._session_history(progress, session_id)
                    total_steps = self.config.simulation.target_turns * 2
                    session_pbar = tqdm(
                        total=total_steps,
                        initial=len(history),
                        desc=self._session_progress_label(session_id),
                        unit="step",
                        position=max(0, int(self.config.runtime.progress_position or 0)),
                        leave=True,
                        dynamic_ncols=True,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    )
                resolved_latent_state = self.client_agent.resolve_latent_state(session_id)

                history = self._session_history(progress, session_id)
                if progress.get("current_session_close_pending") and history and history[-1]["speaker"] == "Client":
                    if session_pbar is not None:
                        session_pbar.close()
                        session_pbar = None
                    self._finalize_session(progress, session_id, resolved_latent_state)
                    self._save_outputs(progress)
                    completed_sessions, total_sessions = self._session_counts(progress)
                    self._log_progress(
                        f"completed session {session_id}/{total_sessions} "
                        f"sessions={completed_sessions}/{total_sessions}"
                    )
                    continue

                speaker, turn_id = self._next_step(progress, session_id)
                refer_list = []
                if history:
                    prior_rows = self._session_rows(progress, session_id)
                    refer_list = [str(prior_rows[-1]["message_id"])]

                query = history[-1]["text"] if history else "session opening and longitudinal continuity"
                search_result = self._search(query=query)
                short_term_note = self._build_carryover_note(progress)
                prompt_retrieved_memories = self._prompt_retrieved_memories(
                    turn_id=turn_id,
                    query=query,
                    session_history=history,
                    search_result=search_result,
                )

                if speaker == "Counselor":
                    prompt_messages = self.counselor_agent.build_prompt_messages(
                        session_id=session_id,
                        turn_id=turn_id,
                        target_turns=self.config.simulation.target_turns,
                        session_history=history,
                        retrieved_memories=prompt_retrieved_memories,
                        short_term_note=short_term_note,
                    )
                    turn = self.counselor_agent.respond(
                        session_id=session_id,
                        turn_id=turn_id,
                        target_turns=self.config.simulation.target_turns,
                        session_history=history,
                        retrieved_memories=prompt_retrieved_memories,
                        short_term_note=short_term_note,
                        temperature=self.config.simulation.counselor_temperature,
                        max_tokens=self._counselor_generation_max_tokens(),
                    )
                    store_response = self._store_message(
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker=speaker,
                        text=turn.message,
                        refer_list=refer_list,
                    )
                    self._record_step(
                        progress=progress,
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker=speaker,
                        text=turn.message,
                        thinking=turn.thinking,
                        query=query,
                        prompt_messages=prompt_messages,
                        search_result=search_result,
                        prompt_retrieved_memories=prompt_retrieved_memories,
                        store_response=store_response,
                        raw_message=turn.raw_message,
                        resolved_latent_state=resolved_latent_state,
                    )
                    if session_pbar is not None:
                        session_pbar.set_postfix_str(f"t{turn_id:03d} Counselor")
                        session_pbar.update(1)
                    if turn.should_end:
                        progress["current_session_close_pending"] = True
                        progress["current_session_end_reason"] = turn.reason
                else:
                    prompt_messages = self.client_agent.build_prompt_messages(
                        counselor_message=history[-1]["text"],
                        session_id=session_id,
                        turn_id=turn_id,
                        target_turns=self.config.simulation.target_turns,
                        session_history=history,
                        retrieved_memories=prompt_retrieved_memories,
                        short_term_note=short_term_note,
                        resolved_latent_state=resolved_latent_state,
                    )
                    response = self.client_agent.respond(
                        counselor_message=history[-1]["text"],
                        session_id=session_id,
                        turn_id=turn_id,
                        target_turns=self.config.simulation.target_turns,
                        session_history=history,
                        retrieved_memories=prompt_retrieved_memories,
                        short_term_note=short_term_note,
                        temperature=self.config.simulation.client_temperature,
                        max_tokens=self._client_generation_max_tokens(),
                        resolved_latent_state=resolved_latent_state,
                    )
                    store_response = self._store_message(
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker="Client",
                        text=response.text,
                        refer_list=refer_list,
                    )
                    self._record_step(
                        progress=progress,
                        session_id=session_id,
                        turn_id=turn_id,
                        speaker="Client",
                        text=response.text,
                        thinking=response.thinking,
                        query=query,
                        prompt_messages=prompt_messages,
                        search_result=search_result,
                        prompt_retrieved_memories=prompt_retrieved_memories,
                        store_response=store_response,
                        raw_message=response.raw_message,
                        resolved_latent_state=resolved_latent_state,
                    )
                    if session_pbar is not None:
                        session_pbar.set_postfix_str(f"t{turn_id:03d} Client")
                        session_pbar.update(1)

                self._save_outputs(progress)

            if session_pbar is not None:
                session_pbar.close()
                session_pbar = None
            progress["status"] = "completed"
            progress["completed_at"] = now_iso()
            self._save_outputs(progress)
            completed_sessions, total_sessions = self._session_counts(progress)
            self._log_progress(f"completed sessions={completed_sessions}/{total_sessions}")
            return self.run_dir
        except Exception as exc:
            if session_pbar is not None:
                session_pbar.close()
                session_pbar = None
            retryable = isinstance(exc, RetryableGenerationError)
            progress["status"] = "running" if retryable else "failed"
            progress["completed_at"] = None if retryable else now_iso()
            failure_payload = {"error": str(exc), "timestamp": now_iso()}
            if retryable:
                failure_payload["retryable"] = True
                retry_after = getattr(exc, "retry_after_seconds", None)
                if retry_after is not None:
                    failure_payload["retry_after_seconds"] = retry_after
            self.failure_writer.write(failure_payload)
            self._save_outputs(progress)
            completed_sessions, total_sessions = self._session_counts(progress)
            self._log_progress(
                f"{'interrupted' if retryable else 'failed'} "
                f"sessions={completed_sessions}/{total_sessions} error={exc}"
            )
            raise
