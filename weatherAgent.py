import json
import os
import requests
import subprocess
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def get_weather(city: str):
    # TODO!: Do an actual API Call
    print("🔨 Tool Called: get_weather", city)
    
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}."
    return "Something went wrong"


def run_command(command: str):
    print("🔨 Tool Called: run_command", command)
    try:
        completed = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        output = completed.stdout.strip() or completed.stderr.strip()
        return output or "Command executed with no output"
    except Exception as exc:
        return f"Command failed: {exc}"

def search_web(query: str):
    print("🔨 Tool Called: search_web", query)
    try:
        # Using Google Custom Search API
        api_key = os.getenv("GOOGLE_API_KEY")
        cx = os.getenv("GOOGLE_CX")
        
        if not api_key or not cx:
            return "Error: GOOGLE_API_KEY and GOOGLE_CX environment variables are required for Google Search API"
        
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={requests.utils.quote(query)}&num=5"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            if items:
                results = []
                for i, item in enumerate(items[:3], 1):  # Limit to top 3 results
                    title = item.get('title', 'No title')
                    snippet = item.get('snippet', 'No snippet')
                    link = item.get('link', 'No link')
                    results.append(f"{i}. {title}\n   {snippet}\n   {link}")
                
                return "\n\n".join(results)
            else:
                return f"No search results found for '{query}'"
        else:
            return f"Search failed with status code: {response.status_code} - {response.text}"
    except Exception as exc:
        return f"Search failed: {exc}"

available_tools = {
    "get_weather": {"fn": get_weather},
    "run_command": {"fn": run_command},
    "search_web": {"fn": search_web},
}

system_prompt = f"""
    You are an helpfull AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    - run_command: Takes a command as input to execute on system and returns ouput
    - search_web: Takes a search query as input and returns real-time Google search results (requires GOOGLE_API_KEY and GOOGLE_CX environment variables)
    
    Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}

    User Query: What is the capital of France?
    Output: {{ "step": "plan", "content": "The user is asking for factual information about France's capital" }}
    Output: {{ "step": "plan", "content": "I should use search_web to get real-time information from Google" }}
    Output: {{ "step": "action", "function": "search_web", "input": "capital of France" }}
    Output: {{ "step": "observe", "output": "1. Paris - Wikipedia\n   Paris is the capital and most populous city of France.\n   https://en.wikipedia.org/wiki/Paris" }}
    Output: {{ "step": "output", "content": "The capital of France is Paris." }}
"""
messages = [
    { "role": "system", "content": system_prompt }
]

while True:
    user_query = input('> ')
    messages.append({ "role": "user", "content": user_query })

    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=messages
        )

        parsed_output = json.loads(response.choices[0].message.content)
        messages.append({ "role": "assistant", "content": json.dumps(parsed_output) })

        if parsed_output.get("step") == "plan":
            print(f"🧠: {parsed_output.get("content")}")
            continue
        
        if parsed_output.get("step") == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")

            if available_tools.get(tool_name, False) is not False:
                output = available_tools[tool_name].get("fn")(tool_input)
                messages.append({ "role": "assistant", "content": json.dumps({ "step": "observe", "output": output}) })
                continue
        
        if parsed_output.get("step") == "output":
            print(f"🤖: {parsed_output.get("content")}")
            break