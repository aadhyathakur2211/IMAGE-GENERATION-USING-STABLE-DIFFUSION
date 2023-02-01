#confirming gpu availability
import os
# os.system("nvidia-smi")
# os.system("pip install diffusers==0.11.0")
# os.system("pip install transformers scipy ftfy")
# os.system('pip install "ipywidgets>=7,<8"') 
# os.system("pip install flask-ngrok")
# os.system("pip install flask==0.12.2")  # Newer versions of flask don't work in Colab
#                             # See https://github.com/plotly/dash/issues/257
# os.system("pip install pyngrok==4.1.1")
# flask_ngrok_example.py
from flask import Flask
#from flask_ngrok import run_with_ngrok
from flask import send_file,render_template,request
import os
import requests


#os.system("ngrok authtoken 2KmcY8Fez0Vt1kGCJbQijgNyRVZ_2HGMfYeYDYm9qpXKr7hcc")

# Enabling widgets in google colab to have notebook_login access
#from google.colab import output
#output.enable_custom_widget_manager()

# Loggin in to HuggingFace using usertoken
from huggingface_hub import notebook_login , login

login('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')

#Upgrading diffusers from 0.3.0 to 0.11.0 to avoid attribute error during stablediffusion pipeline
# os.system("pip install --upgrade diffusers transformers scipy")


# os.system("pip install accelerate")

#Creating StableDiffusionPipeline in HuggingFace
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                               #since google colab has limited computation capacity, we are loading weights from half precision branch
                                               revision = "fp16", torch_dtype = torch.float16, use_auth_token = True)


#Moving our pipeline to GPU for faster inference
pipe = pipe.to("cuda")


#providing text prompt to generate images using stable diffusion, the generated images are in PIL format.
prompt = """a leprachaun with a hat."""
# Since we are using half-precision weights, We will use 'autocast' to run the inference faster 
# from torch import autocast
# with autocast("cuda"):
#    image =pipe(prompt).images[0]
# #display(image)
# image.save(f"panda_surfer.png")


# To have some deterministic output, I have set some random seed to the pipeline.
#generator = torch.Generator("cuda").manual_seed(100)
#with autocast("cuda"):
#  image = pipe(prompt, generator = generator).images[0]
#image
     

# For higher quality images, I also played around with the number of inference steps; more the number_of_steps, better the results
#image = pipe(prompt, num_inference_steps=500).images[0]
#image
     
index = '''
<html>
<head>
    <title>Loading Spinner</title>
    <script>
        function submitForm() {
            document.getElementById("spinner").style.display = "block";
            document.getElementById("form").submit();
        }
    </script>
    <script>
    document.getElementById("myform").addEventListener("keyup", function(event) {
    if (event.keyCode === 13) {
      event.preventDefault();
      document.getElementById("spinner").style.display = "block";
      document.getElementById("form").submit();
    }
    });
    </script>
</head>
<body>
    <form id="form" action="{{url_for('submit')}}" method="post">
        <input type="text" placeholder="Enter some data" name="input_param">
        <input type="text" placeholder="Enter Inference steps numeric (more means more time)" name="inference">
        <button type="button" onclick="submitForm()">Submit</button>
    </form>
    <div id="spinner" style="display: none;">
        Loading...
    </div>
</body>
</html>         
'''

uri = 'http://ww4.sinaimg.cn/large/a7bf601fjw1f7jsbj34a1g20kc0bdnph.gif'
filename = "templates/index.html"
loading_filename = "static/loadingimage.gif"

if os.path.exists(filename):
    os.remove(filename)

os.makedirs(os.path.dirname(filename), exist_ok=True)
os.makedirs(os.path.dirname(loading_filename), exist_ok=True)

with open(loading_filename, 'wb') as f:
  f.write(requests.get(uri).content)

f = open(filename, "a")
f.write(index)
f.close()

app = Flask(__name__)
#run_with_ngrok(app)  # Start ngrok when app is run
#Append preprocessed images in a list


@app.route('/')
def my_form():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    input_param = request.form['input_param']
    inference = request.form['inference']
    # Do some task with input_param here
    import time
    from torch import autocast
    with autocast("cuda"):
      image =pipe(input_param, num_inference_steps=inference).images[0]
    image.save(f"panda_surfer.png")
    return send_file('panda_surfer.png', mimetype='image/png')



if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)  # If address is in use, may need to terminate other sessions:
               # Runtime > Manage Sessions > Terminate Other Sessions
               