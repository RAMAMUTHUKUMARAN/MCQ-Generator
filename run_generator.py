# filepath: c:\Users\Ramamuthukumaran s\OneDrive\Desktop\projects\mcq generator\run_generator.py
from generator import MCQGenerator
import os
# Replace with your actual Gemini API key
API_KEY = os.environ.get("GEMINI_API_KEY")

mcq_gen = MCQGenerator(api_key=API_KEY)
topic = "Photosynthesis"
complexity = 2
history = []

result = mcq_gen.generate_mcq(topic, complexity, history)
mcq_gen.save_mcq_to_txt(result)