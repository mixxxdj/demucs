# Exporting HTDemucs Model

## Convert PyTorch model to ONNX and ORT

- Convert Demucs PyTorch model to ONNX:

```python
python ./scripts/convert-pth-to-onnx.py ./onnx-models
```

- Optionally, convert ONNX model to ORT:

```python
python -m onnxruntime.tools.convert_onnx_models_to_ort ./onnx-models --enable_type_reduction 
```

## Using C++ scripts

- Install dependencies

```git
git submodule update --init --recursive
```

- Set ONNXRuntime path in `./cppscripts/src_cli/CMakeLists.txt`

- Compile the C++ code

```bash
cd cppscripts
make cli
cd ..
```

- Run inference using the following command

```bash
mkdir ./separated/htdemucs_cpp/
./cppscripts/build/build-cli/demucs ./onnx-models/htdemucs.ort ./test.mp3 ./separated/htdemucs_cpp/
```
