import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

from google.genai import types
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools.function_tool import FunctionTool

# ============== ENV & AUTH ==============
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ Add GOOGLE_API_KEY to your .env file")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Retry options
retry_config = types.HttpRetryOptions(
    attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504]
)


# ============== MOCK WEATHER TOOL (Replaces MCP) ==============
def get_current_weather(city: str) -> dict:
    """Mock weather API - returns realistic weather data by city."""
    city_lower = city.lower()

    # Realistic weather patterns by city/keywords
    if any(word in city_lower for word in ['helsinki', 'snow', 'cold']):
        return {"location": city, "condition": "snowy", "temp": "-5°C"}
    elif any(word in city_lower for word in ['miami', 'dubai', 'sunny', 'hot']):
        return {"location": city, "condition": "sunny", "temp": "28°C"}
    elif any(word in city_lower for word in ['london', 'rain', 'seattle']):
        return {"location": city, "condition": "rainy", "temp": "12°C"}
    else:
        return {"location": city, "condition": "clear", "temp": "20°C"}


def get_packing_list(condition: str) -> dict:
    """Packing recommendations based on weather."""
    condition_lower = condition.lower()
    if "rain" in condition_lower or "shower" in condition_lower:
        items = ["Raincoat", "Umbrella", "Waterproof shoes"]
    elif "snow" in condition_lower or "ice" in condition_lower:
        items = ["Heavy coat", "Gloves", "Hat", "Thermal layers"]
    elif "sunny" in condition_lower or "clear" in condition_lower:
        items = ["Sunscreen", "Sunglasses", "Light jacket", "Hat"]
    else:
        items = ["Light layers", "Comfortable shoes", "Travel adapter"]
    return {"status": "success", "recommendations": items}


# ============== AGENT ==============
travel_planner_agent = LlmAgent(
    name="Travel_Planner",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction=(
        "You are a travel planning assistant that helps with packing. "
        "ALWAYS follow this exact process:\n"
        "1. Call 'get_current_weather' FIRST with the exact city name\n"
        "2. Use the 'condition' from weather result in 'get_packing_list'\n"
        "3. Respond with: Weather summary + bullet list of packing items\n"
        "4. Keep responses concise and helpful."
    ),
    tools=[
        FunctionTool(func=get_current_weather),
        FunctionTool(func=get_packing_list),
    ],
)

planner_runner = InMemoryRunner(agent=travel_planner_agent)


async def run_planner_once(city: str) -> str:
    """Run agent and extract final response."""
    try:
        response = await planner_runner.run_debug(
            f"What should I pack for {city}?", verbose=False
        )
        texts = []
        for event in response:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        texts.append(part.text)
        return "\n".join(texts[-2:]) if texts else "No response generated."
    except Exception as e:
        return f"Error: {str(e)}"


# ============== STREAMLIT UI ==============
st.set_page_config(page_title="Travel Planner", page_icon="🧳", layout="wide")
st.title("🧳 Real-time Travel Planner")
st.markdown("Enter a city to get personalized packing recommendations!")

col1, col2 = st.columns([3, 1])
with col1:
    city = st.text_input("🌍 Destination:", placeholder="e.g. Helsinki, Miami, London")
with col2:
    st.info("💡 Try: 'snowy Helsinki' or 'sunny Miami'")

if st.button("✈️ Get Packing List", type="primary") and city.strip():
    with st.spinner("Planning your trip..."):
        answer = asyncio.run(run_planner_once(city.strip()))
    st.success("✅ Your packing list:")
    st.markdown(answer)

st.caption("Powered by Google ADK + Gemini (Mock weather data for demo)")
