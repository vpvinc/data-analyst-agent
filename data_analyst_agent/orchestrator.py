"""Orchestrator that chains data-cleaning and EDA sub-graphs."""

import logging
from typing import Optional, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from data_cleaning_agent import make_lightweight_data_cleaning_agent
from eda_workflow.eda_workflow import make_eda_baseline_workflow

from data_analyst_agent.guardrails import check_pii_columns

logger = logging.getLogger(__name__)
AGENT_NAME = "data_analyst_agent"


class DataAnalystAgent:
    """Thin orchestration layer for data cleaning and EDA.

    Compiles a LangGraph parent graph that runs the cleaning sub-graph
    first and, on success, feeds the cleaned data into the EDA sub-graph.

    Parameters
    ----------
    model : langchain_core.language_models.BaseChatModel
        The chat model used by both sub-graphs.
    checkpointer : object, optional
        LangGraph checkpointer for state persistence.

    Attributes
    ----------
    response : dict or None
        Raw output from the last ``invoke_workflow`` call.
    """

    def __init__(self, model, checkpointer: Optional[object] = None) -> None:
        self.model = model
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_data_analyst_agent(
            model=model,
            checkpointer=checkpointer,
        )

    def invoke_workflow(
        self,
        filepath: str,
        user_instructions: Optional[str] = None,
        max_retries: int = 3,
        retry_count: int = 0,
        **kwargs,
    ) -> None:
        """Read a CSV and run the full cleaning-then-EDA pipeline."""
        df = pd.read_csv(filepath)

        response = self._compiled_graph.invoke(
            {
                "data_raw": df.to_dict(),
                "user_instructions": user_instructions,
                "max_retries": max_retries,
                "retry_count": retry_count,
                "pii_flagged_columns": [],
                "data_cleaned": None,
                "cleaning_response": {},
                "eda_response": {},
            },
            **kwargs,
        )

        self.response = response
        return None

    def get_data_cleaned(self) -> Optional[pd.DataFrame]:
        """Return the cleaned DataFrame, or ``None`` if unavailable."""
        if self.response and self.response.get("data_cleaned"):
            return pd.DataFrame(self.response.get("data_cleaned"))
        return None

    def get_eda_summary(self) -> Optional[str]:
        """Return the EDA summary text, or ``None`` if unavailable."""
        if self.response:
            return self.response.get("eda_response", {}).get("summary")
        return None

    def get_eda_recommendations(self) -> Optional[list]:
        """Return the list of EDA recommendations, or ``None`` if unavailable."""
        if self.response:
            return self.response.get("eda_response", {}).get("recommendations")
        return None

    def get_eda_results(self) -> Optional[dict]:
        """Return the raw EDA results dict, or ``None`` if unavailable."""
        if self.response:
            return self.response.get("eda_response", {}).get("results")
        return None

    def get_pii_flags(self) -> list:
        """Return column names flagged as potential PII, or empty list."""
        if self.response:
            return self.response.get("pii_flagged_columns", [])
        return []


def make_data_analyst_agent(model, checkpointer: Optional[object] = None):
    """Build a parent graph that orchestrates existing cleaning and EDA graphs."""

    # Compile each sub-graph once so they can be invoked as nodes.
    cleaning_graph = make_lightweight_data_cleaning_agent(
        model=model,
        checkpointer=checkpointer,
    )
    eda_graph = make_eda_baseline_workflow(
        model=model,
        checkpointer=checkpointer,
    )

    # Shared state that flows through every node in the parent graph.
    class OrchestrationState(TypedDict):
        data_raw: dict
        user_instructions: Optional[str]
        max_retries: int
        retry_count: int
        pii_flagged_columns: list
        data_cleaned: Optional[dict]
        cleaning_response: dict
        eda_response: dict

    def pii_check_node(state: OrchestrationState) -> dict:
        """Flag columns that look like PII before any LLM call."""
        logger.info("Running PII guardrail")
        columns = list(state.get("data_raw", {}).keys())
        flagged = check_pii_columns(columns)
        if flagged:
            logger.warning("PII guardrail flagged columns: %s", flagged)
        return {"pii_flagged_columns": flagged}

    def route_after_pii_check(state: OrchestrationState) -> str:
        """Block the pipeline if PII columns were detected."""
        if state.get("pii_flagged_columns"):
            return "end"
        return "clean_data"

    def clean_data_node(state: OrchestrationState) -> dict:
        """Invoke the cleaning sub-graph and return cleaned data."""
        logger.info("Running cleaning graph")

        # Map parent state keys to the cleaning sub-graph's expected inputs.
        cleaning_response = cleaning_graph.invoke(
            {
                "user_instructions": state.get("user_instructions"),
                "data_raw": state.get("data_raw", {}),
                "max_retries": state.get("max_retries", 3),
                "retry_count": state.get("retry_count", 0),
            }
        )

        return {
            "data_cleaned": cleaning_response.get("data_cleaned"),
            "cleaning_response": cleaning_response,
        }

    def run_eda_node(state: OrchestrationState) -> dict:
        """Invoke the EDA sub-graph on the cleaned data."""
        logger.info("Running EDA graph")

        # TODO: Invoke eda_graph with the cleaned data and return the response.
        eda_response = eda_graph.invoke(
            {
                "dataframe": state.get("data_cleaned", {}),
            }
        )
        # if results not empty, return the response
        if eda_response.get("results"):
            return {
                "eda_response": eda_response
                }

    def route_after_cleaning(state: OrchestrationState) -> str:
        """Route to EDA if cleaning succeeded, otherwise end."""
        # TODO: Return "run_eda" or "end" based on the cleaning result.
        cleaned = state.get("data_cleaned")
        if cleaned:
            return "run_eda"
        return "end"

    # TODO: Assemble the graph — add nodes, set entry point, and wire edges.
    workflow = StateGraph(OrchestrationState)
    workflow.add_node("pii_check", pii_check_node)
    workflow.add_node("clean_data", clean_data_node)
    workflow.add_node("run_eda", run_eda_node)
    workflow.set_entry_point("pii_check")
    workflow.add_conditional_edges(
        "pii_check",
        route_after_pii_check,
        {"clean_data": "clean_data", "end": END},
    )
    workflow.add_conditional_edges(
        "clean_data",
        route_after_cleaning,
        {"run_eda": "run_eda", "end": END},
    )
    workflow.add_edge("run_eda", END)

    return workflow.compile(checkpointer=checkpointer, name=AGENT_NAME)
