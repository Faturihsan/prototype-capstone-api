const crypto = require('crypto');
const { predictImageSegmentation } = require('../services/inferenceService');
const costPrediction = require('../services/costPrediction')
const InputError = require('../exceptions/InputError');
const { loadObjectDetectionModel, loadRegressionModel } = require("../services/loadModel");


async function postPredictHandler(request, h) {
    const { image } = request.payload;
    const { objectDetectionModel, regressionModel  } = request.server.app.models;

    try {
        const { result, image: base64ImageWithBoundingBoxes, countClasses } = await predictImageSegmentation(objectDetectionModel, image);
        const finalCost =  await costPrediction(regressionModel, countClasses);
        const id = crypto.randomUUID();
        const createdAt = new Date().toISOString();

        const data = {
            id: id,
            finalCost: finalCost,
            result: result,
            image: base64ImageWithBoundingBoxes, // Include the base64 string in the response
            createdAt: createdAt,
        };

        const response = h.response({
            status: "success",
            message: "Object detection is successful",
            data,
        });

        response.code(201);
        console.log('Image processing successful');
        return response;
    } catch (error) {
        console.error('Error processing image:', error);
        throw new InputError('Failed to process image');
    }
}

module.exports = postPredictHandler;
