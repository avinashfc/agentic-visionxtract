from crewai import Agent, LLM
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from crewai_workflow.agents.extractor_agent.tools.extract_data import extract_signature_card_data

load_dotenv()


class ExtractorAgent:
    def __init__(self):
        self.config = self._load_config()
        # Temporarily unset GOOGLE_CLOUD_PROJECT to force API key usage instead of Vertex AI
        original_project = os.environ.pop('GOOGLE_CLOUD_PROJECT', None)
        original_use_vertexai = os.environ.pop('GOOGLE_GENAI_USE_VERTEXAI', None)
        try:
            print(f"Loading Extractor Agent model: {self.config['model']}")
            self.llm = LLM(
                model=self.config['model'],
                api_key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'),
                temperature=0.0,
                timeout=60
            )
        finally:
            # Restore environment variables if they were set
            if original_project is not None:
                os.environ['GOOGLE_CLOUD_PROJECT'] = original_project
            if original_use_vertexai is not None:
                os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = original_use_vertexai  

    def _load_config(self):
        with open(Path(__file__).parent / 'config' / 'agent_config.yaml', 'r') as file:
            return yaml.safe_load(file)
        
    def create_agent(self):
        return Agent(
            name=self.config['name'],
            role=self.config['role'],
            goal=self.config['goal'],
            backstory=self.config['backstory'],
            verbose=self.config['verbose'],
            allow_delegation=self.config['allow_delegation'],
            max_iter=self.config['max_iter'],
            max_execution_time=90,
            tools=[extract_signature_card_data],
            llm=self.llm
        )

