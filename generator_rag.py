import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def generate_mcq_from_context(api_key, context, topic, complexity):
    prompt = PromptTemplate(
        template=(
            "Using the following context from a PDF:\n"
            "{context}\n\n"
            "Generate a unique multiple-choice question about {topic} with complexity level {complexity}.\n"
            "Present the output as a valid JSON object with keys: question, options (list of 4), answer (A/B/C/D), explanation.\n"
            "Options must be plausible and only one correct. Use double quotes for JSON. No extra text."
        ),
        input_variables=["context", "topic", "complexity"]
    )
    llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-2.0-flash", temperature=0.3)
    _input = prompt.format(context=context[:4000], topic=topic, complexity=complexity)
    output = llm.invoke(_input)
    import re, json
    content = output.content if hasattr(output, "content") else output
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        json_str = match.group(0)
        data = json.loads(json_str)
        # If options is a string, convert to list
        options = data.get("options")
        if isinstance(options, str):
            options_list = re.findall(r'"(.*?)"', options)
            data["options"] = options_list if options_list else [options]
        return data
    else:
        raise ValueError("Could not parse MCQ from model output.")

class MCQApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF MCQ Generator (Gemini RAG)")
        self.api_key = "AIzaSyBlVjr8l9d58nzmD1F4YLr17Qa1oTEty3s"
        if not self.api_key:
            messagebox.showerror("API Key Error", "GEMINI_API_KEY environment variable not set.")
            root.destroy()
            return

        self.pdf_path = ""
        self.pdf_text = ""
        self.setup_gui()

    def setup_gui(self):
        frm = tk.Frame(self.root)
        frm.pack(padx=10, pady=10)

        tk.Button(frm, text="Select PDF", command=self.select_pdf).grid(row=0, column=0, sticky="ew")
        self.topic_entry = tk.Entry(frm, width=40)
        self.topic_entry.grid(row=0, column=1, padx=5)
        self.topic_entry.insert(0, "Enter topic (e.g. Photosynthesis)")

        tk.Label(frm, text="Complexity (1-5):").grid(row=0, column=2)
        self.complexity_var = tk.IntVar(value=2)
        tk.Spinbox(frm, from_=1, to=5, textvariable=self.complexity_var, width=5).grid(row=0, column=3)

        tk.Label(frm, text="Number of Questions:").grid(row=0, column=4)
        self.num_questions_var = tk.IntVar(value=1)
        tk.Spinbox(frm, from_=1, to=10, textvariable=self.num_questions_var, width=5).grid(row=0, column=5)

        tk.Button(frm, text="Generate MCQ(s)", command=self.generate_mcqs).grid(row=0, column=6, padx=5)

        self.output_box = scrolledtext.ScrolledText(self.root, width=100, height=20)
        self.output_box.pack(padx=10, pady=10)

    def select_pdf(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path:
            self.pdf_path = path
            self.pdf_text = extract_pdf_text(path)
            messagebox.showinfo("PDF Loaded", f"Loaded PDF: {os.path.basename(path)}")

    def generate_mcqs(self):
        if not self.pdf_text:
            messagebox.showerror("No PDF", "Please select a PDF first.")
            return
        topic = self.topic_entry.get().strip()
        if not topic or topic.startswith("Enter topic"):
            messagebox.showerror("No Topic", "Please enter a topic.")
            return
        complexity = self.complexity_var.get()
        num_questions = self.num_questions_var.get()
        self.output_box.delete(1.0, tk.END)
        history = set()
        for i in range(num_questions):
            try:
                mcq = generate_mcq_from_context(self.api_key, self.pdf_text, topic, complexity)
                # Avoid duplicate questions
                if mcq.get('question') in history:
                    continue
                history.add(mcq.get('question'))
                self.display_mcq(mcq, i+1)
            except Exception as e:
                self.output_box.insert(tk.END, f"Error generating question {i+1}: {str(e)}\n")

    def display_mcq(self, mcq, number):
        self.output_box.insert(tk.END, f"MCQ {number}:\n")
        self.output_box.insert(tk.END, f"Question: {mcq.get('question')}\n")
        options = mcq.get('options', [])
        for opt in options:
            self.output_box.insert(tk.END, f"  {opt}\n")
        self.output_box.insert(tk.END, f"Answer: {mcq.get('answer')}\n")
        self.output_box.insert(tk.END, f"Explanation: {mcq.get('explanation')}\n")
        self.output_box.insert(tk.END, "-" * 40 + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = MCQApp(root)
    root.mainloop()