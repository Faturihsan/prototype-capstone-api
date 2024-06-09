const ort = require("onnxruntime-node");
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

async function loadObjectDetectionModel() {
    const modelPath = path.resolve(__dirname, '../../models/object-detection/best.onnx');
    const model = await ort.InferenceSession.create(modelPath);
    return model;
}

async function loadRegressionModel() {
    // const modelPath = path.resolve(__dirname, '../../models/model-regression/model.json');
    const modelPath = 'file://' + path.resolve(__dirname, '../../models/model-regression/model.json');
    const model = await tf.loadGraphModel(modelPath);
    return model;
    // const modelUrl = "file://models/model-regression/model.json";
    // return tf.loadGraphModel(modelUrl);
}

module.exports = {loadObjectDetectionModel, loadRegressionModel}; 