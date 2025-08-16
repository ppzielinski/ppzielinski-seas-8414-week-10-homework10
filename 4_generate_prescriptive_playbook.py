# Filename: 4_generate_prescriptive_playbook.py
import json
import asyncio
import aiohttp
import os
import sys


async def generate_playbook(xai_findings, api_key):
    """
    Creates a simple, step-by-step incident response playbook using the Gemini API.
    """
    prompt = f"""
    As a SOC Manager, your task is to create a simple, step-by-step incident response playbook for a Tier 1 analyst.
    The playbook should be based on the provided alert details and the explanation from our AI model.

    Do not explain the AI model; only provide the prescriptive actions. The playbook must be a numbered list of 3-4 clear, concise steps.

    **Alert Details & AI Explanation:**
    {xai_findings}
    """

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(apiUrl, json=payload) as response:
                result = await response.json()

                if response.status != 200:
                    return f"Error: API returned status {response.status}. Response: {json.dumps(result)}"

                if result.get('candidates'):
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "Error: Could not generate playbook. Response was: " + json.dumps(result)

    except aiohttp.ClientConnectorError as e:
        return f"An error occurred: Could not connect to the API endpoint. {e}"
    except Exception as e:
        return f"An error occurred: {e}"


# Data representing an alert from a DGA detection model
findings = """- **Alert:** Potential DGA domain detected in DNS logs.
- **Domain:** `kq3v9z7j1x5f8g2h.info`
- **Source IP:** `10.1.1.50` (Workstation-1337)
- **AI Model Explanation (from SHAP):** The model flagged this domain with 99.8% confidence primarily due to its very high character entropy and long length, which are strong indicators of an algorithmically generated domain."""


async def main():
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("---")
        print("🚨 Error: GOOGLE_API_KEY environment variable not set.")
        print("To run this script, you need to set your API key.")
        print("\nFor Linux/macOS, use:\n  export GOOGLE_API_KEY='YOUR_API_KEY_HERE'")
        print("\nFor Windows (PowerShell), use:\n  $env:GOOGLE_API_KEY=\"YOUR_API_KEY_HERE\"")
        print("\nReplace 'YOUR_API_KEY_HERE' with the key you obtained from Google AI Studio.")
        print("---")
        sys.exit(1)

    # --- Displaying context and input data ---
    print("---")
    print("Context: Generating a prescriptive playbook from alert findings.")
    print("Input being sent to Gemini:")
    print(findings)
    print("--------------------------------------------------")
    print("\n--- AI-Generated Playbook ---")
    # -------------------------------------------

    playbook = await generate_playbook(findings, api_key)
    print(playbook)


if __name__ == "__main__":
    asyncio.run(main())
