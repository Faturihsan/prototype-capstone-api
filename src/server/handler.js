const crypto = require('crypto');
const { predictImageSegmentation } = require('../services/inferenceService');
const InputError = require('../exceptions/InputError'); // Assuming InputError is defined in exceptions


async function postPredictHandler(request, h) {
    const { image } = request.payload;
    const { model } = request.server.app;

    try {
        console.log('Processing image...');
        const result = await predictImageSegmentation(model, image);
        
        const id = crypto.randomUUID();
        const createdAt = new Date().toISOString();

        const data = {
            id: id,
            result: result,
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
