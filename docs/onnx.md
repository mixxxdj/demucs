# Exporting HTDemucs Model

## Convert PyTorch model to ONNX and ORT

- Convert Demucs PyTorch model to ONNX:

```python
python ./scripts/convert-pth-to-onnx.py ./onnx-models
```

- Then, convert ONNX to ORT:

```bash
./scripts/convert-model-to-ort.sh ./onnx-models
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
