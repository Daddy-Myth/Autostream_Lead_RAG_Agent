"""
AutoStream Conversational AI Agent
Social-to-Lead Agentic Workflow using LangGraph + Ollama (Llama 3.1)
"""

import json
import re
from pathlib import Path
from typing import Annotated, TypedDict, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────
# 1.  KNOWLEDGE BASE  (RAG – local JSON)
# ─────────────────────────────────────────────

KB_PATH = Path(__file__).parent / "knowledge_base.json"

def load_knowledge_base() -> str:
    """Load and format the knowledge base into a readable string for the LLM."""
    with open(KB_PATH) as f:
        kb = json.load(f)

    lines = [
        f"Product: {kb['product_name']} — {kb['tagline']}",
        "",
        "== PRICING PLANS ==",
    ]
    for plan in kb["plans"]:
        lines.append(f"\n{plan['name']} ({plan['price']}):")
        for feat in plan["features"]:
            lines.append(f"  • {feat}")

    lines += ["", "== COMPANY POLICIES =="]
    for p in kb["policies"]:
        lines.append(f"  • {p}")

    lines += ["", "== FAQs =="]
    for faq in kb["faqs"]:
        lines.append(f"  Q: {faq['question']}")
        lines.append(f"  A: {faq['answer']}")

    return "\n".join(lines)


KNOWLEDGE_BASE = load_knowledge_base()


# ─────────────────────────────────────────────
# 2.  MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Simulate saving a lead to a CRM / backend."""
    print("\n" + "═" * 50)
    print("🎯  LEAD CAPTURED SUCCESSFULLY")
    print(f"   Name     : {name}")
    print(f"   Email    : {email}")
    print(f"   Platform : {platform}")
    print("═" * 50 + "\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ─────────────────────────────────────────────
# 3.  AGENT STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    intent: str                                # casual | inquiry | high_intent
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool                      # True when we're in lead-collection mode


# ─────────────────────────────────────────────
# 4.  LLM  (Ollama – Llama 3.1 8B)
# ─────────────────────────────────────────────

llm = ChatOllama(model="llama3.1", temperature=0.3)


# ─────────────────────────────────────────────
# 5.  SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are Alex, a friendly and knowledgeable sales assistant for AutoStream — an AI-powered video editing SaaS for content creators.

Your job:
1. Answer product questions accurately using ONLY the knowledge base below.
2. Detect when users show high buying intent (e.g. "I want to sign up", "I want to try", "I'm ready", "let's do it", "how do I get started").
3. When high intent is detected, collect: Name, Email, and Creator Platform (YouTube, Instagram, TikTok, etc.) — one at a time, politely.
4. Be concise, warm, and helpful. Never make up features or pricing.

== KNOWLEDGE BASE ==
{KNOWLEDGE_BASE}

== INTENT CLASSIFICATION RULES ==
- casual       : greetings, small talk, vague questions
- inquiry      : questions about pricing, features, policies, comparisons
- high_intent  : user explicitly wants to sign up, try, buy, or get started

When collecting lead info, ask for one piece at a time:
  Step 1 → Ask for their name
  Step 2 → Ask for their email
  Step 3 → Ask for their content platform

Do NOT confirm lead capture in your text response — the system handles that separately.
Once all three are collected, simply say: "LEAD_READY" on its own line at the very end of your response (hidden from user).
"""


# ─────────────────────────────────────────────
# 6.  HELPER: DETECT INTENT
# ─────────────────────────────────────────────

def detect_intent(user_message: str, current_intent: str) -> str:
    """
    Simple keyword-based intent detection.
    Once high_intent is reached it stays high_intent.
    """
    if current_intent == "high_intent":
        return "high_intent"

    msg = user_message.lower()

    high_intent_keywords = [
        "sign up", "signup", "subscribe", "want to try", "want to buy",
        "i'm ready", "im ready", "let's go", "lets go", "get started",
        "i want the", "i'll take", "purchase", "checkout", "enroll",
        "start my", "create my account", "register", "i'm in", "im in"
    ]
    inquiry_keywords = [
        "price", "pricing", "cost", "plan", "feature", "refund", "support",
        "resolution", "video", "caption", "how does", "what is", "tell me",
        "explain", "difference", "compare", "trial", "cancel", "discount"
    ]

    if any(k in msg for k in high_intent_keywords):
        return "high_intent"
    if any(k in msg for k in inquiry_keywords):
        return "inquiry"
    return "casual"


# ─────────────────────────────────────────────
# 7.  HELPER: EXTRACT LEAD FIELDS FROM REPLY
# ─────────────────────────────────────────────

def extract_lead_fields(state: AgentState, user_msg: str) -> AgentState:
    """
    During lead collection, extract name / email / platform
    from the user's latest message based on what's still missing.
    """
    msg = user_msg.strip()

    if not state.get("lead_name"):
        state["lead_name"] = msg
    elif not state.get("lead_email"):
        # Validate basic email format
        if re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", msg):
            state["lead_email"] = msg
        else:
            state["lead_email"] = msg   # store anyway; LLM will re-ask if wrong
    elif not state.get("lead_platform"):
        state["lead_platform"] = msg

    return state


# ─────────────────────────────────────────────
# 8.  LANGGRAPH NODES
# ─────────────────────────────────────────────

def chat_node(state: AgentState) -> AgentState:
    """Core chat node: calls LLM with full message history."""

    # Build message list for LLM
    system = SystemMessage(content=SYSTEM_PROMPT)
    history = state["messages"]

    response = llm.invoke([system] + history)
    reply_text = response.content

    # Check if LLM signalled all lead info is collected
    lead_ready = "LEAD_READY" in reply_text
    clean_reply = reply_text.replace("LEAD_READY", "").strip()

    new_state = dict(state)
    new_state["messages"] = [AIMessage(content=clean_reply)]

    if lead_ready and not state.get("lead_captured"):
        # Fire the mock tool
        result = mock_lead_capture(
            name=state.get("lead_name", ""),
            email=state.get("lead_email", ""),
            platform=state.get("lead_platform", ""),
        )
        new_state["lead_captured"] = True
        new_state["collecting_lead"] = False

    return new_state


def intent_node(state: AgentState) -> AgentState:
    """Detect intent and decide whether to start collecting lead info."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    new_intent = detect_intent(last_human, state.get("intent", "casual"))
    new_state = dict(state)
    new_state["intent"] = new_intent

    # Start lead collection if high intent and not already collecting / captured
    if (
        new_intent == "high_intent"
        and not state.get("collecting_lead")
        and not state.get("lead_captured")
    ):
        new_state["collecting_lead"] = True

    return new_state


def lead_extraction_node(state: AgentState) -> AgentState:
    """Extract lead fields from user message when in collecting mode.
    
    We only extract AFTER the agent has already asked for a field,
    i.e. when collecting_lead was already True BEFORE this turn.
    The intent_node sets collecting_lead=True on the high-intent turn,
    but we must not extract on that same turn (the user hasn't answered yet).
    We detect this by checking: if collecting_lead just became True this turn,
    none of the lead fields will be set yet AND the last human message is the
    high-intent sentence — skip extraction for that turn.
    """
    if not state.get("collecting_lead") or state.get("lead_captured"):
        return state

    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    # Count how many AI messages have been sent since collecting started.
    # If the agent hasn't asked for anything yet (0 AI messages after intent),
    # this is the trigger turn — don't extract yet.
    has_name = bool(state.get("lead_name"))
    has_email = bool(state.get("lead_email"))
    has_platform = bool(state.get("lead_platform"))

    # If nothing collected yet, check whether the last human message looks like
    # a name/answer or like a high-intent sentence (contains "sign up", "pro plan" etc.)
    if not has_name:
        high_intent_markers = [
            "sign up", "signup", "subscribe", "want to try", "want to buy",
            "i'm ready", "im ready", "let's go", "lets go", "get started",
            "i want the", "i'll take", "purchase", "checkout", "enroll",
            "start my", "create my account", "register", "i'm in", "im in"
        ]
        # If the message that triggered high_intent is still the last human message,
        # skip — the agent hasn't asked for the name yet.
        if any(k in last_human.lower() for k in high_intent_markers):
            return state  # don't extract on the trigger message

    return extract_lead_fields(dict(state), last_human)


# ─────────────────────────────────────────────
# 9.  ROUTING
# ─────────────────────────────────────────────

def route(state: AgentState) -> str:
    """Always go to chat after intent + extraction nodes."""
    return "chat"


# ─────────────────────────────────────────────
# 10.  BUILD GRAPH
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("extract", lead_extraction_node)
    graph.add_node("chat", chat_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "extract")
    graph.add_conditional_edges("extract", route, {"chat": "chat"})
    graph.add_edge("chat", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 11.  MAIN LOOP
# ─────────────────────────────────────────────

def main():
    print("━" * 55)
    print("  AutoStream AI Agent  |  Powered by Llama 3.1 + LangGraph")
    print("  Type 'exit' or 'quit' to end the conversation.")
    print("━" * 55 + "\n")

    app = build_graph()

    state: AgentState = {
        "messages": [],
        "intent": "casual",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Agent: Thanks for chatting! Have a great day 🎬")
            break

        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run graph
        state = app.invoke(state)

        # Print last AI reply
        last_ai = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            ""
        )
        print(f"\nAgent: {last_ai}\n")

        # Show intent badge (optional debug info)
        intent_badge = {
            "casual": "💬 casual",
            "inquiry": "🔍 inquiry",
            "high_intent": "🔥 high intent",
        }.get(state.get("intent", "casual"), "")
        print(f"[Intent: {intent_badge}]\n")


if __name__ == "__main__":
    main()
