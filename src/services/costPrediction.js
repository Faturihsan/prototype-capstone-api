const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');
const fs = require('fs');
const path = require('path');
const { predictImageSegmentation } = require('../services/inferenceService');
const scalerParams = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../models/model-regression/scaler.json'), 'utf8'));
const yScalerParams = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../models/model-regression/package.json'), 'utf8'));

// console.log(scalerParams)
function standardScaler(data, mean, std) {
    return data.map(row => row.map((x, i) => (x - mean[i]) / std[i]));
}

function minMaxScaler(data, min, max) {
    return data.map(row => row.map(x => x * (max - min) + min));
}

async function costPrediction (model, countClasses){
    try{
        const input = [countClasses]

        const inputData = standardScaler(input, scalerParams.mean, scalerParams.std);
        const inputTensor = tf.tensor2d(inputData);
        const costPrediction = await model.predict(inputTensor).array()
        
        const output = minMaxScaler(costPrediction, yScalerParams.min[0], yScalerParams.max[0]);
        const formattedOutput = output.map(value => `Rp. ${Math.round(value[0])}`);
        return formattedOutput;

    }catch{
        throw new InputError(`Terjadi kesalahan input: ${error.message}`);
    }
}

module.exports = costPrediction;