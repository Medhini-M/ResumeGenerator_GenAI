import os
import torch
from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, Gemma3ForCausalLM

# load model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-pt")  
model = Gemma3ForCausalLM.from_pretrained("google/gemma-3-27b-pt")   

app = Flask(__name__)   # Flask constructor 

# function to check if information is valid
def validate(info, pipeln):
    prompt=f"""
Check if the below data is about a human's professional occupation. Give 'yes' if it is relevant, else 'no'.

User Info:
{info}

Result: 
"""
    output = pipeln(prompt)[0]["generated_text"]
    res = output[len(prompt):].strip()
    return res


# function to generate resume
def generate_detailed_resume(minimal_info):
    resume_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0,             # -1 for CPU, 0 for GPU
        max_new_tokens=400,   #800
        do_sample=True,       #creativity
        temperature=0.7,
        top_p=0.9
    )

    res = validate(minimal_info, resume_generator)    #pass input, pipeline
    #print("Before: ", res)
    if (res.lower()).startswith('yes'):
        res = 1      
    else:
        res = 0
    #print(res)
    
    # only if information is valid, pass prompt for resume
    if res==1:
        prompt = f"""
    You are an expert resume writer. Using the information below, write a complete, professional resume without repeating any lines. Include the following sections:
    
    - Personal Information
    Include name, profession, professional email and phone number.
    
    - Education
    Include university name, degree, and graduation year based on experience.
    
    - Work Experience
    Include job title, company name, dates, and 3-4 bullet points describing responsibilities and achievements. Be creative and realistic.
    
    - Skills
    List relevant technical and soft skills.
    
    - Projects
    Describe 1-2 relevant projects using the mentioned skills. Mention project name, description and skills used.
    
    User Info:
    {minimal_info}
    
    Resume:
    """
        output = resume_generator(prompt)[0]["generated_text"]
        #del resume_generator
        #torch.cuda.empty_cache()
        #print("Clearing cache")
        gen_output = output[len(prompt):].strip()
        return gen_output
    else:
        return "null"


# function to execute prompts 
def call_Gemma(info):
    resume = generate_detailed_resume(info)
    if resume!="null":
        return "This is your generated resume: \n\n"+resume+"\n\n"
    else:
        return "Your information cannot be used to generate a resume. Kindly provide the necessary details.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"



# A decorator used to tell the application which URL is associated function 
# decorator to route URL 
@app.route("/") 
# binding to the function of route 
def form():
    return render_template("form.html")

@app.route("/result", methods=['GET', 'POST'])
def send():
    data = request.args.get('q')
    #print("/result?q="+data)
    return call_Gemma(data)

# run flask application
if __name__=="__main__": 
   app.run()

   