# Restaurant Ordering Agent

An AI agent that takes real food orders through natural conversation — built to demonstrate a different category of "agentic AI" than a typical Q&A chatbot: an agent that **takes real, transactional actions** on a business's behalf, not just answers questions.

## The business pitch

A customer opens a chat (WhatsApp-style, web, or app) and orders naturally — "I'll have 2 zinger burgers and a fries" — the same way they'd talk to a waiter. The agent:

- Understands the order, item by item, even when multiple items are mentioned in one message
- Tracks a running cart across the whole conversation
- Answers questions about the menu or the current total
- **Places a real, saved order** once the customer confirms — a structured record with their name, phone number, exact items, and total, ready for restaurant staff to fulfill

**The pitch to a restaurant owner:** your staff no longer needs to personally take every order by phone or WhatsApp. The AI handles that conversation end-to-end, 24/7, and hands staff a clean, already-correct order to prepare — not a chat log they have to re-read and interpret.

## Why this is a different kind of agent than a RAG chatbot

| | Typical RAG chatbot | This agent |
|---|---|---|
| Purpose | Answer questions | Complete a transaction |
| State | Conversation memory only | Conversation memory **+ a running cart** |
| Tools | Read-only (search, retrieve) | **Write actions** with real side effects (saving an order) |
| Output | An answer | **A saved order record**, ready for a human to act on |

## The reliability engineering story

Early versions of this agent had a serious, business-critical bug: when a customer mentioned multiple items in one message ("2 zinger burgers **and** a fries"), the AI model would sometimes silently drop one of them — a real restaurant would have lost that item from the order entirely, with no indication anything went wrong.

This was fixed with **defense in depth**, not just a better prompt:

1. **A deterministic pre-parser** scans every customer message for menu items using plain text matching (not AI) — before the AI model even sees the message. This guarantees every mentioned item is caught, regardless of how the AI model behaves.
2. **A multi-round tool-calling loop**, so the AI can take multiple sequential actions in one turn if needed, rather than being cut off after a single tool call.
3. **A code-level duplicate guard** — discovered that even after removing a tool from what's offered to the model, this local model sometimes tried calling it anyway. Rather than trusting that restriction alone, the code also explicitly blocks and no-ops any duplicate `add_to_cart` call for items the pre-parser already handled.
4. **Defensive argument handling** — tool calls with missing or malformed arguments (e.g. a missing quantity) default sensibly instead of crashing the whole request.

This was verified by checking the actual saved order data (`restaurant_orders.json`), not just trusting the AI's own conversational summary of what it had done — the two didn't always match, which was itself an important finding.

## Architecture

```
Customer message
      ↓
Deterministic pre-parser (catches every mentioned item, guaranteed)
      ↓
LLM (handles natural conversation, menu questions, confirmations)
      ↓
Tool execution (add_to_cart / view_cart / place_order) — with duplicate guards
      ↓
Saved order record (JSON file here; MongoDB/Firebase in production)
```

## Tech stack

- **LLM:** Qwen2.5-VL-3B-Instruct, running locally and for free via [LM Studio](https://lmstudio.ai)
- **Backend:** Python, `openai` client pointed at the local LM Studio server
- **Order storage:** JSON file for this demo — designed to swap in MongoDB or Firebase for production

## Running it locally

1. Open LM Studio, load a tool-calling-capable model, and start the local server
2. Install dependencies: `pip install openai`
3. Run: `python restaurant_agent.py` — this runs a full simulated ordering conversation automatically

## What I learned building this

- Multi-item requests are a genuine reliability gap for small local models — they can silently drop items even when explicitly instructed not to
- Restricting which tools are offered to a model is not always sufficient to prevent it from trying to call a tool anyway — a code-level execution guard is a stronger guarantee
- For business-critical correctness (orders, money), deterministic code should handle what absolutely cannot be wrong, while the LLM handles the parts that benefit from natural conversation
- Always verify against the actual saved/persisted data, not the AI's own summary of what it did — the two can silently diverge

## Possible next steps

- Wrap this as a FastAPI service with a simple chat frontend, so it's demoable to a real business
- Swap the JSON file for MongoDB or Firebase
- Add order status tracking (received → preparing → ready)
- Support order modifications after initial confirmation ("actually remove the fries")
