import os
import gradio as gr
import requests
import inspect
import pandas as pd
from smolagents import tool, CodeAgent, OpenAIServerModel, DuckDuckGoSearchTool
from tools import read_image, transcribe_audio, run_video, search_wikipedia, read_code

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
model_id = "gpt-4.1"

# --- Basic Agent Definition ---
class BasicAgent:
    def __init__(self, model_id=model_id):
        model = OpenAIServerModel(model_id=model_id, temperature=0.1)
        duckduck_search = DuckDuckGoSearchTool()
        self.agent = CodeAgent(
            model = model,
            tools = [read_image, transcribe_audio, read_code, run_video, search_wikipedia, duckduck_search],
            additional_authorized_imports = ["numpy", "pandas"],
            max_steps = 20
            )
        add_sys_prompt = f"""\n\nIf a file_url is available or an url is given in question statement, then request and use the content to answer the question. \
        If a code file, such as .py file, is given, do not attempt to execute it but rather open it as a text file and analyze the content. \
        When a tabluar file, such as csv, tsv, xlsx, is given, read it using pandas. 
        
        Make sure you provide the answer in accordance with the instruction provided in the question. Do not return the result of tool as a final_answer. 
        Do Not add any additional information, explanation, unnecessary words or symbols. The answer is likely as simple as one word."""
        self.agent.prompt_templates['system_prompt'] += add_sys_prompt

    def __call__(self, question: str) -> str:
        answer = self.agent.run(question)
        return answer


def run_all():
    """
    Fetches all questions, runs the BasicAgent on them, and displays the results.
    """

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Initialize Agent
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name")
        if file_name:
            file_url = f"{DEFAULT_API_URL}/files/{task_id}"
        else:
            file_url = "No URL provided"        
        extension = file_name.split(".")[-1]
        
        ## Augment question_text with file_url and file_extension
        question_text += f"\n\nfile_url : {file_url} \nfile_extension : {extension}"
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    return results_log

if __name__ == "__main__":
    results_log = run_all()
    
