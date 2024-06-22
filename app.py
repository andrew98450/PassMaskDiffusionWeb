import onnxruntime
import numpy
import json
import turbo_flask
import flask
import threading
import time

charmap = json.load(open("./password_passmaskldiffusion_charmap.json", "r"))

app = flask.Flask(__name__)
turbo = turbo_flask.Turbo(app)
t_step = 50
seq_len = 20

decoder_session = onnxruntime.InferenceSession("./decoder_mask_model.onnx", providers=["CPUExecutionProvider"])
denoise_session = onnxruntime.InferenceSession("./ldiffusion_mask_model.onnx", providers=["CPUExecutionProvider"])

softmax = lambda x: numpy.exp(x) / numpy.sum(numpy.exp(x), axis=-1, keepdims=True)

def preprocessing(x, batch_size, seq_len):
    text_data = [charmap['stoi'][chars] if chars != "'" else charmap['stoi']["<m>"] for chars in str(x).strip()]
    text_data.insert(0, charmap['stoi']["<s>"])
    while len(text_data) < seq_len:
        text_data.append(charmap['stoi']["<p>"])
    inputs = [text_data[:seq_len] for _ in range(batch_size)]
    inputs = numpy.array(inputs, dtype=numpy.int64)
    return inputs

def sample(text : str, sample_size = 25):
    x = []
    xt = numpy.random.standard_normal((sample_size, 64)).astype(numpy.float32)
    c = preprocessing(text, sample_size, seq_len)
    betas = numpy.linspace(0.001, 0.2, num=t_step, dtype=numpy.float32)

    alpha = 1 - betas
    alpha_hat = numpy.cumprod(alpha, axis=0)
    sigma = numpy.sqrt(betas)
    for i in reversed(range(t_step)):
        z = numpy.random.standard_normal((sample_size, 64)).astype(numpy.float32) if i > 0 else numpy.zeros_like(xt, dtype=numpy.float32)
        t = numpy.full((sample_size,), i, dtype=numpy.int64)

        lambdas = (1 - alpha[i]) / numpy.sqrt(1 - alpha_hat[i])
        outputs = denoise_session.run(
            None, 
            input_feed={"inputs": xt, "t": t, "c": c})[0]
        xt = (1 / numpy.sqrt(alpha[i])) * (xt - (lambdas * outputs)) + (z * sigma[i])
        x.append(softmax(decoder_session.run(None, input_feed={"inputs": xt})[0]))
        
    return x

def process_diffusion(cond_text, n):
    with app.app_context():
        x = sample(cond_text, n)
        for t in range(t_step):
            outputs = x[t]
            outputs = numpy.argmax(outputs, axis=-1)
            passwords = []
            for i in range(n):
                text = ""
                for j in range(seq_len):
                    if charmap["itos"][outputs[i, j]] == "<p>":
                        break
                    text += charmap["itos"][outputs[i, j]]
                passwords.append(text)
            turbo.push(
                turbo.update(
                    flask.render_template("process.html", passwords=passwords, t=t+1, range=[i for i in range(len(passwords))]), target="diffusion_process"))
            time.sleep(0.1)

@app.route("/", methods=["GET", "POST"])
def index():
    if flask.request.method == "POST":
        n = int(flask.request.form["n"])
        cond_text = str(flask.request.form["cond_text"])
        thread = threading.Thread(target=process_diffusion, args=(cond_text, n))
        thread.start()
    return flask.render_template("index.html")

app.run(host="0.0.0.0", port=8080)