# bias_detector/model_loader.py
# Handles loading different model types automatically based on file extension.
# Supports scikit-learn (.joblib), ONNX (.onnx), TensorFlow (.h5, SavedModel),
# and PyTorch (.pt). Optional dependencies are only required if that model
# type is actually used.

import os

# try importing each optional library
# if not installed, we set a flag and handle it gracefully later

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class SklearnModelWrapper:
    """Wraps a scikit-learn model so it has a standard predict() interface."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)


class ONNXModelWrapper:
    """Wraps an ONNX model so it has a standard predict() interface."""

    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def predict(self, X):
        import numpy as np
        # ONNX expects float32 numpy arrays
        X_array = np.array(X, dtype=np.float32)
        result = self.session.run(None, {self.input_name: X_array})
        # result[0] contains the predictions
        return result[0]


class TensorFlowModelWrapper:
    """Wraps a TensorFlow/Keras model so it has a standard predict() interface."""

    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold  # converts probabilities to 0/1

    def predict(self, X):
        import numpy as np
        X_array = np.array(X, dtype=np.float32)
        probabilities = self.model.predict(X_array)
        # convert probabilities to binary predictions
        return (probabilities >= self.threshold).astype(int).flatten()


class PyTorchModelWrapper:
    """Wraps a PyTorch model so it has a standard predict() interface."""

    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.model.eval()  # set to evaluation mode

    def predict(self, X):
        import torch
        import numpy as np
        X_tensor = torch.tensor(np.array(X, dtype=np.float32))
        with torch.no_grad():
            output = self.model(X_tensor)
        # convert output to binary predictions
        probabilities = torch.sigmoid(output).numpy().flatten()
        return (probabilities >= self.threshold).astype(int)


def load_model(model_path):
    """
    Loads a model from disk based on file extension.

    Supported formats:
      .joblib  -> scikit-learn model
      .onnx    -> ONNX model (requires: pip install onnxruntime)
      .h5      -> TensorFlow/Keras model (requires: pip install tensorflow)
      .pt      -> PyTorch model (requires: pip install torch)

    For SavedModel format (TensorFlow directory), pass the folder path.

    Returns a wrapped model with a standard predict(X) interface.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: '{model_path}'\n"
            f"Check that the path in config.json is correct."
        )

    # ── scikit-learn .joblib ───────────────────────────────────────────────────
    if model_path.endswith(".joblib"):
        if not JOBLIB_AVAILABLE:
            raise ImportError(
                "joblib is not installed.\n"
                "Run: pip install joblib"
            )
        model = joblib.load(model_path)
        print(f"  Loaded scikit-learn model: {type(model).__name__}")
        return SklearnModelWrapper(model)

    # ── ONNX .onnx ────────────────────────────────────────────────────────────
    elif model_path.endswith(".onnx"):
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX model detected but onnxruntime is not installed.\n"
                "Run: pip install onnxruntime"
            )
        session = ort.InferenceSession(model_path)
        print(f"  Loaded ONNX model: {model_path}")
        return ONNXModelWrapper(session)

    # ── TensorFlow/Keras .h5 ──────────────────────────────────────────────────
    elif model_path.endswith(".h5"):
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow/Keras model detected but tensorflow is not installed.\n"
                "Run: pip install tensorflow"
            )
        model = tf.keras.models.load_model(model_path)
        print(f"  Loaded TensorFlow/Keras model: {model_path}")
        return TensorFlowModelWrapper(model)

    # ── PyTorch .pt ───────────────────────────────────────────────────────────
    elif model_path.endswith(".pt"):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch model detected but torch is not installed.\n"
                "Run: pip install torch"
            )
        model = torch.load(model_path, map_location="cpu")
        print(f"  Loaded PyTorch model: {model_path}")
        return PyTorchModelWrapper(model)

    # ── TensorFlow SavedModel directory ───────────────────────────────────────
    elif os.path.isdir(model_path):
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow SavedModel detected but tensorflow is not installed.\n"
                "Run: pip install tensorflow"
            )
        model = tf.saved_model.load(model_path)
        print(f"  Loaded TensorFlow SavedModel: {model_path}")
        return TensorFlowModelWrapper(model)

    # ── Unknown format ─────────────────────────────────────────────────────────
    else:
        extension = os.path.splitext(model_path)[1]
        raise ValueError(
            f"Unsupported model format: '{extension}'\n"
            f"Supported formats: .joblib, .onnx, .h5, .pt, or a TensorFlow SavedModel directory."
        )