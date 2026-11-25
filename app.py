import onnxruntime
import numpy
import json
import flask
import threading # Needed for the original index route if kept

# Load configuration and models
charmap = json.load(open("./password_passmaskldiffusion_charmap.json", "r"))

decoder_session = onnxruntime.InferenceSession("./decoder_mask_model.onnx", providers=["CPUExecutionProvider"])
denoise_session = onnxruntime.InferenceSession("./ldiffusion_mask_model.onnx", providers=["CPUExecutionProvider"])

t_step = 50
seq_len = 20
default_sample_count = 10
max_sample_count = 100

# Initialize Flask app and Turbo (if needed for progress tracking, though we will focus on API now)
app = flask.Flask(__name__)

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

def process_diffusion(cond_text, n, on_step=None):
    # Removed app.app_context() as it's not strictly necessary for a standalone API call context if we are just running the logic.
    x = sample(cond_text, n)
    passwords = []
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
        if on_step is not None:
            on_step(passwords, t)
    return passwords

def push_progress(passwords, t):
    # Keeping this function definition but removing the actual call to keep it clean for API focus if needed later.
    pass 


# --- API Endpoints following openai.yaml structure ---

@app.route("/sample", methods=["GET"])
def get_samples():
    """
    Operation: getSamples
    Summary: Generate samples
    Description: Generate N samples based on mask_text.
    Parameters:
      - name: mask_text (query, required)
      - name: N (query, optional)
        Type: integer
        Default: 10
        Minimum: 1
        Maximum: 100
    Responses:
      200: Successful response (application/json)
      400: Invalid request (application/json)
    """
    mask_text = flask.request.args.get("mask_text")
    if mask_text is None or not mask_text.strip():
        return flask.jsonify({"error": "mask_text is required"}), 400

    n_str = flask.request.args.get("N", str(default_sample_count))
    try:
        n = int(n_str)
    except (TypeError, ValueError):
        return flask.jsonify({"error": "N must be an integer"}), 400

    if n < 1 or n > max_sample_count:
        return flask.jsonify(
            {"error": f"N must be between 1 and {max_sample_count}"}), 400

    passwords = process_diffusion(mask_text, n)
    return flask.jsonify({"data": passwords})


# Removed the root route (@app.route("/", methods=["GET", "POST"])) to focus purely on API endpoints.

if __name__ == "__main__":
    # Running in production/API mode
    app.run(host="0.0.0.0", port=8080)