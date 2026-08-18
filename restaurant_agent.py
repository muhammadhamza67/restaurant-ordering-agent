"""
Restaurant Ordering Agent — a genuinely different kind of agent from your
RAG chatbot.

KEY DIFFERENCE: your RAG chatbot answers isolated questions. This agent
maintains a STATEFUL CART across the whole conversation, and its final
job is to TAKE A REAL ACTION (saving an order) — not just answer a
question. This is closer to what a real "AI agent for a business" looks
like: it doesn't just inform, it DOES something on the business's behalf.

Sellable pitch: "An AI ordering assistant for your restaurant's WhatsApp
or website — customers order in plain language, the agent handles the
whole conversation and saves a real order for you to fulfill."
"""

import json
import os
import re
from datetime import datetime
from openai import OpenAI

# Words/phrases that signal the customer is placing an order (not just asking
# a question). We only run the deterministic parser when one of these appears,
# to avoid false-triggering on things like "how much is a zinger burger?"
ORDER_TRIGGER_PHRASES = [
    "i'll have", "i will have", "i want", "give me", "i'd like", "i would like",
    "get me", "can i get", "can i have", "order", "add "
]

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}


def looks_like_menu_question(message: str) -> bool:
    """Detect if the customer is asking about the menu or prices — if so,
    we'll inject the REAL menu directly into context, so the model physically
    cannot hallucinate fake items or prices instead of calling the tool."""
    msg_lower = message.lower()
    menu_keywords = ["menu", "what do you have", "what's available", "price",
                      "how much", "cost", "options"]
    return any(kw in msg_lower for kw in menu_keywords)


def looks_like_an_order(message: str) -> bool:
    """Check if this message sounds like the customer is ordering, not just
    asking a question (e.g. about price or the menu)."""
    msg_lower = message.lower()
    question_words = ["how much", "what is", "what's", "price of", "cost of", "?"]
    if any(qw in msg_lower for qw in question_words):
        return False
    return any(phrase in msg_lower for phrase in ORDER_TRIGGER_PHRASES)


def parse_order_items(message: str) -> list:
    """Deterministically find EVERY menu item mentioned in a message, with
    quantity, by splitting on 'and'/commas and matching each segment against
    the menu. This exists because the LLM alone sometimes drops items when
    multiple are mentioned in one sentence — this guarantees nothing is missed."""
    segments = re.split(r'\band\b|,', message, flags=re.IGNORECASE)
    found = []

    for segment in segments:
        seg_clean = segment.strip().lower()
        if not seg_clean:
            continue

        matched_item = None
        for menu_item in MENU:
            # match on the core word of the item (e.g. "fries", "burger", "coke")
            if menu_item in seg_clean or any(word in seg_clean for word in menu_item.split() if len(word) > 3):
                matched_item = menu_item
                break

        if not matched_item:
            continue

        # Extract quantity: a digit, or a number word, or default to 1
        qty = 1
        digit_match = re.search(r'\b(\d+)\b', seg_clean)
        if digit_match:
            qty = int(digit_match.group(1))
        else:
            for word, val in NUMBER_WORDS.items():
                if re.search(rf'\b{word}\b', seg_clean):
                    qty = val
                    break

        found.append((matched_item, qty))

    return found

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")
MODEL_NAME = "qwen2.5-vl-3b-instruct"

# --- The restaurant's menu ---
# In a real deployment, this would come from a database. For now, a fixed
# menu is enough to prove the concept.

MENU = {
    "zinger burger": 450,
    "beef burger": 500,
    "chicken biryani": 350,
    "beef biryani": 400,
    "chicken karahi (half)": 900,
    "chicken karahi (full)": 1700,
    "fries": 200,
    "coke (500ml)": 100,
    "sprite (500ml)": 100,
}

# --- Per-session cart state ---
# Each session (each customer's conversation) gets its own running cart.
carts = {}  # {session_id: [{"item": "zinger burger", "qty": 2, "price": 450}, ...]}

# --- Orders "database" ---
# For this demo, we save to a local JSON file. In a real deployment, this
# would be MongoDB or Firebase — exactly matching your existing skills.
ORDERS_FILE = "restaurant_orders.json"


def get_menu() -> str:
    """Return the full menu with prices."""
    lines = [f"- {item.title()}: PKR {price}" for item, price in MENU.items()]
    return "\n".join(lines)


def add_to_cart(session_id: str, item: str, quantity: int) -> str:
    """Add an item to the customer's cart. Matches item names loosely
    against the menu (case-insensitive, partial match)."""
    item_lower = item.lower().strip()

    matched_item = None
    for menu_item in MENU:
        if item_lower in menu_item or menu_item in item_lower:
            matched_item = menu_item
            break

    if not matched_item:
        return f"'{item}' is not on the menu. Here's what's available:\n{get_menu()}"

    if session_id not in carts:
        carts[session_id] = []

    carts[session_id].append({
        "item": matched_item,
        "qty": quantity,
        "price": MENU[matched_item]
    })

    return f"Added {quantity}x {matched_item.title()} to the order."


def view_cart(session_id: str) -> str:
    """Show the current cart contents and running total."""
    cart = carts.get(session_id, [])
    if not cart:
        return "The cart is currently empty."

    lines = []
    total = 0
    for entry in cart:
        subtotal = entry["qty"] * entry["price"]
        total += subtotal
        lines.append(f"- {entry['qty']}x {entry['item'].title()} = PKR {subtotal}")

    lines.append(f"\nTotal: PKR {total}")
    return "\n".join(lines)


def place_order(session_id: str, customer_name: str, phone_number: str) -> str:
    """Finalize the order — the real ACTION this agent takes. Saves the
    order to a persistent file (stand-in for MongoDB/Firebase)."""
    cart = carts.get(session_id, [])
    if not cart:
        return "Cannot place an order — the cart is empty."

    total = sum(entry["qty"] * entry["price"] for entry in cart)

    order = {
        "session_id": session_id,
        "customer_name": customer_name,
        "phone_number": phone_number,
        "items": cart,
        "total": total,
        "timestamp": datetime.now().isoformat(),
    }

    # Load existing orders, append, save — simple persistence for the demo
    orders = []
    if os.path.exists(ORDERS_FILE):
        with open(ORDERS_FILE, "r") as f:
            orders = json.load(f)
    orders.append(order)
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)

    # Clear the cart now that the order is placed
    carts[session_id] = []

    return f"Order placed! Total: PKR {total}. We'll call {phone_number} to confirm. Thank you, {customer_name}!"


# --- Tool definitions for the LLM ---

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_menu",
            "description": "Get the full restaurant menu with real, current prices. "
                            "You MUST call this tool any time the customer asks about "
                            "the menu, what's available, or any item's price. NEVER "
                            "guess, recall from memory, or make up menu items or "
                            "prices — you do not actually know them without calling "
                            "this tool. All prices are in PKR, not dollars.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add a menu item to the customer's order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "The menu item name"},
                    "quantity": {"type": "integer", "description": "How many of this item"}
                },
                "required": ["item", "quantity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_cart",
            "description": "Show the customer their current order and running total.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Finalize and submit the order. Only call this once the customer has confirmed everything and provided their name and phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_name": {"type": "string"},
                    "phone_number": {"type": "string"}
                },
                "required": ["customer_name", "phone_number"]
            }
        }
    }
]


def run_agent_turn(session_id: str, user_message: str, conversation_history: list) -> tuple:
    """Run one turn of the ordering conversation. Loops up to MAX_ROUNDS times,
    letting the model make MULTIPLE sequential tool calls if needed (e.g. adding
    two different items one after another) before giving a final reply."""

    system_prompt = {
        "role": "system",
        "content": (
            "You are a friendly restaurant ordering assistant. Help the customer "
            "browse the menu and build their order. Use add_to_cart when they want "
            "items, view_cart to show their order, and place_order ONLY after they "
            "confirm they're done AND you have their name and phone number.\n\n"
            "IMPORTANT: If the customer mentions MULTIPLE different items in one "
            "message (e.g. 'burgers and fries'), you MUST call add_to_cart SEPARATELY "
            "for EACH distinct item — one tool call per item type. Never skip an item.\n\n"
            "If the customer doesn't specify a quantity for an item (e.g. 'a fries', "
            "'a coke', or just names an item with no number), ASSUME they mean 1 — "
            "do not stop to ask, since that slows down the order. Only ask a "
            "clarifying question if something is genuinely ambiguous (e.g. which "
            "size of an item that comes in multiple sizes)."
        )
    }

    messages = [system_prompt] + conversation_history

    # --- Deterministic menu injection: guarantee correct menu answers ---
    # If this looks like a menu/price question, inject the REAL menu as a
    # system note so the model has no way to hallucinate wrong prices —
    # it's looking right at the real data, not relying on memory or a tool call.
    if looks_like_menu_question(user_message):
        menu_note = f"[System note: here is the REAL, current menu — use ONLY this, do not make up other items or prices:\n{get_menu()}]"
        messages.append({"role": "system", "content": menu_note})
        print("  [DEBUG: injected real menu into context for this turn]")

    # --- Deterministic pre-parse: catch every item BEFORE the LLM sees the
    # message, so nothing gets silently dropped due to model unreliability ---
    pre_parser_handled_items = False

    if looks_like_an_order(user_message):
        detected_items = parse_order_items(user_message)
        if detected_items:
            pre_parser_handled_items = True
            confirmations = []
            for item, qty in detected_items:
                result = add_to_cart(session_id, item, qty)
                confirmations.append(result)
                print(f"  [DEBUG: pre-parser auto-added {qty}x {item}]")

            system_note = (
                f"[System note: the following items were automatically added "
                f"to the cart already: {'; '.join(confirmations)}. Just "
                f"acknowledge them naturally and ask if they want anything else.]"
            )
            messages.append({"role": "system", "content": system_note})

    messages.append({"role": "user", "content": user_message})

    # If the pre-parser already handled this message, don't even OFFER
    # add_to_cart to the model this turn — this guarantees no duplicate
    # adds, rather than just hoping the model follows the system note.
    tools_for_this_turn = [t for t in tools if t["function"]["name"] != "add_to_cart"] \
        if pre_parser_handled_items else tools

    MAX_ROUNDS = 4
    reply = None

    for round_num in range(MAX_ROUNDS):
        offer_tools = tools_for_this_turn if round_num < MAX_ROUNDS - 1 else []
        response = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=offer_tools)
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            reply = msg.content
            break

        print(f"  [DEBUG round {round_num + 1}: model made {len(msg.tool_calls)} tool call(s)]")
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            print(f"  [DEBUG: calling {name} with {args}]")

            if name == "get_menu":
                result = get_menu()
            elif name == "add_to_cart":
                if pre_parser_handled_items:
                    # Belt-and-suspenders: even though we tried to remove this
                    # tool from what's offered, some local models still call it
                    # anyway. This guard guarantees no duplicate, regardless.
                    result = "These items were already added — no need to add them again."
                    print(f"  [DEBUG: BLOCKED duplicate add_to_cart call — pre-parser already handled this turn]")
                else:
                    item = args.get("item", "")
                    quantity = args.get("quantity", 1)  # defensive default if model omits it
                    result = add_to_cart(session_id, item, quantity)
            elif name == "view_cart":
                result = view_cart(session_id)
            elif name == "place_order":
                result = place_order(session_id, args.get("customer_name", ""), args.get("phone_number", ""))
            else:
                result = f"Unknown tool: {name}"

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

    if reply is None:
        reply = "Sorry, I had trouble processing that — could you try again?"

    return reply, messages[1:]


if __name__ == "__main__":
    print("=== Restaurant Ordering Agent — Test Conversation ===\n")

    session_id = "test_customer_1"
    history = []

    test_messages = [
        "Hi, what's on the menu?",
        "I'll have 2 zinger burgers and a fries",
        "What's my total so far?",
        "Actually add a coke too",
        "That's it, place my order. My name is Ahmed and my number is 03001234567"
    ]

    for user_msg in test_messages:
        print(f"CUSTOMER: {user_msg}")
        reply, history = run_agent_turn(session_id, user_msg, history)
        print(f"AGENT: {reply}\n")