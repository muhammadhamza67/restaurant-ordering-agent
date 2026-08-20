"""
WhatsApp integration for the restaurant ordering agent, using Twilio.

HOW THIS WORKS:
1. A customer sends a WhatsApp message to your Twilio sandbox number
2. Twilio receives it and forwards it to YOUR server, by calling the
   /whatsapp endpoint below (this is called a "webhook")
3. Your server runs the message through the SAME restaurant agent you
   already built and tested — nothing about the agent logic changes
4. Your server sends the agent's reply back through Twilio, which
   delivers it to the customer's WhatsApp as a normal message

This means: the exact same reliability fixes (pre-parser, duplicate
guard, menu injection) apply automatically here too — WhatsApp is just
a new "front door" to the same agent.
"""

from fastapi import FastAPI, Form
from fastapi.responses import PlainTextResponse
from twilio.twiml.messaging_response import MessagingResponse
from restaurant_agent import run_agent_turn

app = FastAPI()

# Per-customer conversation history, keyed by their WhatsApp number.
# This is exactly the same pattern as session_id in your other servers —
# here, the phone number itself IS the session ID, which makes sense
# since each WhatsApp number is naturally one customer.
conversations = {}


@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),   # Twilio sends the sender's number as "From"
    Body: str = Form(...)    # Twilio sends the message text as "Body"
):
    """This is the URL Twilio calls every time a WhatsApp message arrives."""

    customer_number = From  # e.g. "whatsapp:+923001234567"
    customer_message = Body

    print(f"\n=== WhatsApp message from {customer_number} ===")
    print(f"Message: {customer_message}")

    # Run the SAME agent logic as before, using the phone number as session_id
    history = conversations.get(customer_number, [])
    reply, updated_history = run_agent_turn(customer_number, customer_message, history)
    conversations[customer_number] = updated_history

    print(f"Reply: {reply}")

    # Twilio expects a specific XML-ish response format (called TwiML)
    # to know what to send back to the customer.
    twiml_response = MessagingResponse()
    twiml_response.message(reply)

    return PlainTextResponse(content=str(twiml_response), media_type="application/xml")


@app.get("/")
def health_check():
    return {"status": "WhatsApp restaurant agent webhook is running"}