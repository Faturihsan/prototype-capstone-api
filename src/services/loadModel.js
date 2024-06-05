const ort = require("onnxruntime-node");
const path = require('path');

async function loadModel() {
    const modelPath = path.resolve(__dirname, '../../models/best.onnx');
    const model = await ort.InferenceSession.create(modelPath);
    return (model);
}

module.exports = loadModel; 