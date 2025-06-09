import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

class LlamaService:
    def __init__(self):
        self.api_token = os.getenv('HUGGINGFACE_API_TOKEN')
        self.api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        self.headers = {"Authorization": f"Bearer {self.api_token}"}
    
    def query_llama(self, prompt):
        """Query the Llama model with a given prompt"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '').strip()
                return result.get('generated_text', '').strip()
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Error querying Llama API: {e}")
            return None
    
    def get_deforestation_solutions(self, location_info="", severity="moderate"):
        """Get solutions for deforestation based on detected patterns"""
        prompt = f"""
        As an environmental expert, provide practical solutions for addressing deforestation. 
        The detected deforestation severity is: {severity}
        Location context: {location_info}
        
        Please provide:
        1. Immediate actions (2-3 solutions)
        2. Long-term strategies (2-3 solutions)
        3. Community involvement approaches (2 solutions)
        4. Technology-based solutions (2 solutions)
        
        Keep responses concise and actionable. Format as clear bullet points.
        """
        
        return self.query_llama(prompt)
    
    def get_image_analytics(self, deforestation_detected, confidence_score=0.0):
        """Get detailed analytics about the deforestation detection"""
        status = "detected" if deforestation_detected else "not detected"
        
        prompt = f"""
        As a forest conservation analyst, provide insights about this satellite image analysis:
        
        Deforestation status: {status}
        Model confidence: {confidence_score:.2f}
        
        Please provide:
        1. Environmental impact assessment (2-3 points)
        2. Potential causes of deforestation in this area (2-3 points)
        3. Biodiversity implications (2 points)
        4. Climate change connections (2 points)
        5. Monitoring recommendations (2 points)
        
        Keep responses scientific but accessible. Use bullet points for clarity.
        """
        
        return self.query_llama(prompt)
    
    def get_prevention_strategies(self, region_type="tropical"):
        """Get prevention strategies based on region type"""
        prompt = f"""
        As a conservation strategist, suggest prevention methods for {region_type} forest regions:
        
        Provide:
        1. Policy recommendations (3 points)
        2. Economic incentives (2 points)
        3. Technological monitoring (2 points)
        4. Community engagement (2 points)
        
        Focus on evidence-based, implementable solutions. Use clear bullet points.
        """
        
        return self.query_llama(prompt)