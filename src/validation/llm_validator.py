"""
LLM-based Validator using LangGraph and Groq.

Implements a 3-node agentic validation pipeline:
1. Extractor Node — Summarizes extracted P&ID data
2. Validator Node — LLM checks engineering consistency
3. Reporter Node — Generates human-readable validation report

Uses Groq's Llama 3.3 70B for fast, free inference.
"""

import logging
import os
from typing import Annotated, Dict, List, TypedDict

from config.settings import settings

logger = logging.getLogger(__name__)


# ---- State Schema ----

class ValidationState(TypedDict):
    """State that flows through the LangGraph validation pipeline."""

    # Input data
    extraction_summary: str
    entity_count: int
    graph_summary: dict
    rule_based_issues: list

    # LLM outputs
    llm_analysis: str
    llm_issues: list
    final_report: str

    # Control
    validation_passed: bool


class LLMValidator:
    """
    LangGraph-based LLM validator for P&ID extraction results.

    Uses a 3-node graph:
    [Summarize] → [Validate] → [Report]
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Args:
            api_key: Groq API key. Falls back to settings/env.
            model: Groq model name.
        """
        self.api_key = api_key or settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or settings.groq_model
        self.graph = None
        self._llm = None

    def _init_llm(self):
        """Initialize the Groq LLM."""
        if not self.api_key:
            logger.warning(
                "No Groq API key configured. LLM validation will be skipped. "
                "Set GROQ_API_KEY in .env or get a free key from console.groq.com"
            )
            return False

        try:
            from langchain_groq import ChatGroq

            self._llm = ChatGroq(
                model=self.model,
                api_key=self.api_key,
                temperature=settings.groq_temperature,
                max_tokens=2048,
            )
            logger.info(f"Groq LLM initialized: {self.model}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            return False

    def _build_graph(self):
        """Build the LangGraph validation pipeline."""
        from langgraph.graph import StateGraph, START, END

        builder = StateGraph(ValidationState)

        # Add nodes
        builder.add_node("summarize", self._summarize_node)
        builder.add_node("validate", self._validate_node)
        builder.add_node("report", self._report_node)

        # Define edges
        builder.add_edge(START, "summarize")
        builder.add_edge("summarize", "validate")
        builder.add_edge("validate", "report")
        builder.add_edge("report", END)

        self.graph = builder.compile()
        logger.info("LangGraph validation pipeline built")

    def validate(
        self,
        entities_data: List[dict],
        graph_summary: dict,
        rule_based_issues: List[dict],
    ) -> dict:
        """
        Run the full LLM validation pipeline.

        Args:
            entities_data: List of entity dicts from extraction.
            graph_summary: Graph structure summary.
            rule_based_issues: Issues from deterministic rule checks.

        Returns:
            Validation result dict with analysis, issues, and report.
        """
        if not self._init_llm():
            return self._fallback_report(rule_based_issues)

        if self.graph is None:
            self._build_graph()

        # Prepare extraction summary for the LLM
        extraction_summary = self._format_extraction_summary(
            entities_data, graph_summary
        )

        # Initial state
        initial_state = ValidationState(
            extraction_summary=extraction_summary,
            entity_count=len(entities_data),
            graph_summary=graph_summary,
            rule_based_issues=rule_based_issues,
            llm_analysis="",
            llm_issues=[],
            final_report="",
            validation_passed=True,
        )

        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            return {
                "llm_analysis": result["llm_analysis"],
                "llm_issues": result["llm_issues"],
                "report": result["final_report"],
                "passed": result["validation_passed"],
            }
        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return self._fallback_report(rule_based_issues)

    # ---- Graph Nodes ----

    def _summarize_node(self, state: ValidationState) -> dict:
        """Node 1: Summarize the extraction data for LLM analysis."""
        prompt = f"""You are an engineering P&ID (Piping & Instrumentation Diagram) validation expert.

Below is a summary of data extracted from a P&ID drawing using AI (YOLOv8 for symbol detection + PaddleOCR for text).
Analyze this extraction and identify any potential issues or inconsistencies.

## Extracted Data Summary
{state['extraction_summary']}

## Graph Structure
- Total nodes: {state['graph_summary'].get('total_nodes', 0)}
- Total edges: {state['graph_summary'].get('total_edges', 0)}
- Entity types: {state['graph_summary'].get('entity_types', {})}
- Connected components: {state['graph_summary'].get('connected_components', 0)}

## Rule-Based Issues Already Found
{self._format_issues(state['rule_based_issues'])}

Provide a concise technical analysis of the extraction quality. Focus on:
1. Completeness — are typical P&ID components present?
2. Consistency — do tag patterns, line numbers, and instrument loops make sense?
3. Potential errors — any obviously incorrect classifications or tags?
"""
        response = self._llm.invoke(prompt)
        return {"llm_analysis": response.content}

    def _validate_node(self, state: ValidationState) -> dict:
        """Node 2: Validate engineering consistency using LLM reasoning."""
        prompt = f"""Based on the following analysis of P&ID extraction data, identify specific engineering validation issues.

## Previous Analysis
{state['llm_analysis']}

For each issue found, provide:
- Issue description
- Severity (error/warning/info)
- Affected entity or component
- Recommended action

Format your response as a numbered list of issues. If no additional issues are found beyond rule-based checks, state "No additional issues found."

Focus only on engineering-relevant issues, NOT on AI/ML model accuracy.
"""
        response = self._llm.invoke(prompt)

        # Parse LLM issues (simple extraction)
        llm_issues = []
        lines = response.content.strip().split("\n")
        current_issue = ""
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                if current_issue:
                    llm_issues.append(current_issue.strip())
                current_issue = line
            elif current_issue:
                current_issue += " " + line

        if current_issue:
            llm_issues.append(current_issue.strip())

        passed = not any("error" in issue.lower() for issue in llm_issues)

        return {
            "llm_issues": llm_issues,
            "validation_passed": passed,
        }

    def _report_node(self, state: ValidationState) -> dict:
        """Node 3: Generate a human-readable validation report."""
        prompt = f"""Generate a concise, professional P&ID extraction validation report.

## Extraction Summary
- Total entities extracted: {state['entity_count']}
- Graph: {state['graph_summary'].get('total_nodes', 0)} nodes, {state['graph_summary'].get('total_edges', 0)} edges

## AI Analysis
{state['llm_analysis']}

## LLM-Identified Issues
{chr(10).join(state['llm_issues']) if state['llm_issues'] else 'None'}

## Rule-Based Issues
{self._format_issues(state['rule_based_issues'])}

Generate a report with:
1. **Overall Assessment** — Pass/Fail with brief justification
2. **Key Findings** — Top 3-5 most important findings
3. **Recommendations** — Actionable next steps
4. **Confidence Level** — Your assessment of extraction quality (High/Medium/Low)

Keep the report under 300 words. Use markdown formatting.
"""
        response = self._llm.invoke(prompt)
        return {"final_report": response.content}

    # ---- Helpers ----

    @staticmethod
    def _format_extraction_summary(entities_data: List[dict], graph_summary: dict) -> str:
        """Format entity data as a readable summary for the LLM."""
        # Group by type
        type_groups: Dict[str, list] = {}
        for entity in entities_data:
            etype = entity.get("entity_type", "unknown")
            if etype not in type_groups:
                type_groups[etype] = []
            type_groups[etype].append(entity)

        lines = []
        for etype, entities in type_groups.items():
            lines.append(f"\n### {etype.title()} ({len(entities)} items)")
            for e in entities[:10]:  # Limit to first 10 per type
                tag = e.get("tag_id", "no-tag")
                desc = e.get("description", "")
                conf = e.get("confidence", 0)
                lines.append(f"  - {tag}: {desc} (confidence: {conf:.2f})")
            if len(entities) > 10:
                lines.append(f"  ... and {len(entities) - 10} more")

        return "\n".join(lines)

    @staticmethod
    def _format_issues(issues: List[dict]) -> str:
        """Format rule-based issues for LLM context."""
        if not issues:
            return "No rule-based issues found."

        lines = []
        for i, issue in enumerate(issues[:15], 1):
            severity = issue.get("severity", "info")
            message = issue.get("message", "Unknown issue")
            lines.append(f"{i}. [{severity.upper()}] {message}")

        if len(issues) > 15:
            lines.append(f"... and {len(issues) - 15} more issues")

        return "\n".join(lines)

    @staticmethod
    def _fallback_report(rule_based_issues: List[dict]) -> dict:
        """Generate a basic report when LLM is unavailable."""
        error_count = sum(1 for i in rule_based_issues if i.get("severity") == "error")
        warning_count = sum(1 for i in rule_based_issues if i.get("severity") == "warning")

        report = f"""# P&ID Extraction Validation Report

## Overall Assessment
**LLM validation unavailable** — showing rule-based results only.

## Rule-Based Findings
- **Errors:** {error_count}
- **Warnings:** {warning_count}
- **Total issues:** {len(rule_based_issues)}

## Recommendations
1. Configure a Groq API key for full LLM-powered validation
2. Review entities flagged with low confidence scores
3. Verify duplicate tags and orphan nodes manually

*Note: Set GROQ_API_KEY in your .env file for AI-powered validation.*
"""
        return {
            "llm_analysis": "LLM unavailable",
            "llm_issues": [],
            "report": report,
            "passed": error_count == 0,
        }
