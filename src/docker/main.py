import os
import http.server
import socketserver
from typing import Optional

PORT = 8000

# Heavy ML imports are optional at edit-time and in environments without the packages;
# guard them so the module can be linted/edited without installing large deps.
try:  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # noqa: BLE001 - broad except for optional imports
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:  # type: ignore
    from optimum.intel.openvino import OVModelForCausalLM  # type: ignore
except Exception:  # noqa: BLE001
    OVModelForCausalLM = None  # type: ignore


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"llama3 container running\n")
        else:
            return super().do_GET()


# Default model name and save dir used only when compression is explicitly requested
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_SAVE_DIR = "/app/compressed_model"


def download_model(model_name: str):
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError("transformers not installed")
    print(f"Downloading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def compress_model(model, save_dir: str):
    if OVModelForCausalLM is None:
        raise RuntimeError("optimum[openvino] not installed")
    print("Compressing model with OpenVINO...")
    ov_model = OVModelForCausalLM.from_pretrained(model, export=True)
    ov_model.save_pretrained(save_dir)
    print(f"Compressed model saved to {save_dir}")


def main(download_and_compress: Optional[bool] = False, save_dir: str = DEFAULT_SAVE_DIR):
    # Optionally download & compress the model when the container is started with that intent.
    if download_and_compress:
        os.makedirs(save_dir, exist_ok=True)
        model, tokenizer = download_model(MODEL_NAME)
        compress_model(model, save_dir)

    # Start a tiny HTTP server to keep the container running and provide a health endpoint.
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"Serving on port {PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    # default: do not download huge models during container start
    main(download_and_compress=False)