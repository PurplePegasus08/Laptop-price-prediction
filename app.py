from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model (pipeline) and dataset (df)
pipe = pickle.load(open(r'D:\model\venv\pipe.pkl', 'rb'))  # This is the model pipeline
df = pickle.load(open(r'D:\model\venv\data.pkl', 'rb'))  # Dataset

# Preparing options for the form
brands = df['Company'].unique().tolist()
types = df['TypeName'].unique().tolist()
ram_options = [2, 4, 6, 8, 12, 16, 24, 32, 64]
hdd_options = [0, 128, 256, 512, 1024, 2048]
ssd_options = [0, 8, 128, 256, 512, 1024]
resolutions = ['1920x1080', '1366x768', '1600x900', '3840x2160', 
               '3200x1800', '2880x1800', '2560x1600', 
               '2560x1440', '2304x1440']
cpus = df['Cpu_brand'].unique().tolist()
gpus = df['Gpu_Brand'].unique().tolist()
os_options = df['OS'].unique().tolist()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', 
                           brands=brands, 
                           types=types, 
                           ram_options=ram_options, 
                           hdd_options=hdd_options, 
                           ssd_options=ssd_options, 
                           resolutions=resolutions, 
                           cpus=cpus, 
                           gpus=gpus, 
                           os_options=os_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        company = request.form['company']
        type_name = request.form['type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = request.form['touchscreen']
        ips = request.form['ips']
        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']
        
        # Encoding categorical features to numeric
        touchscreen = 1 if touchscreen == 'Yes' else 0
        ips = 1 if ips == 'Yes' else 0

        # Convert screen resolution and compute PPI
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        # Create the feature array
        query = np.array([company, type_name, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = np.array(query, dtype=object).reshape(1, -1)

        # Predict the price using the model
        predicted_price = np.exp(pipe.predict(query)[0])

        return render_template('index.html', 
                               brands=brands, 
                               types=types, 
                               ram_options=ram_options, 
                               hdd_options=hdd_options, 
                               ssd_options=ssd_options, 
                               resolutions=resolutions, 
                               cpus=cpus, 
                               gpus=gpus, 
                               os_options=os_options, 
                               predicted_price=int(predicted_price))

    except Exception as e:
        return render_template('index.html', 
                               brands=brands, 
                               types=types, 
                               ram_options=ram_options, 
                               hdd_options=hdd_options, 
                               ssd_options=ssd_options, 
                               resolutions=resolutions, 
                               cpus=cpus, 
                               gpus=gpus, 
                               os_options=os_options, 
                               error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
