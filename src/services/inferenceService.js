const sharp = require('sharp');
const ort = require('onnxruntime-node');

const yolo_classes =  [
    'bonnet_dent', 'bumper_dent', 'bumper_scratch', 'car_window_damage', 'crack', 
    'door_dent', 'door_scratch', 'front_windscreen_damage', 'headlight_damage', 
    'quarter_panel_dent', 'quarter_panel_scratch', 'rear_windscreen_damage', 
    'taillight_damage', 'tire_flat', 'trunk_door_dent'
];

async function preprocessingImage(image) {
    const img = sharp(image);
    const md = await img.metadata();
    const [img_width,img_height] = [md.width, md.height];
    const pixels = await img.removeAlpha()
        .resize({width:640,height:640,fit:'fill'})
        .raw()
        .toBuffer();

    const red = [], green = [], blue = [];
    for (let index=0; index<pixels.length; index+=3) {
        red.push(pixels[index]/255.0);
        green.push(pixels[index+1]/255.0);
        blue.push(pixels[index+2]/255.0);
    }

    const input = [...red, ...green, ...blue];
    return [input, img_width, img_height];
} 

async function run_model(model, input) {
    try {
        console.log('Running model inference...');
        input = new ort.Tensor(Float32Array.from(input), [1, 3, 640, 640]);
        const outputs = await model.run({ images: input });

        console.log('Model inference successful');
        console.log(outputs.output1)
        return outputs.output0.data;
    } catch (error) {
        console.error('Error running model inference:', error);
        throw error;
    }
}



function process_output(output, img_width, img_height) {
    let boxes = [];
    for (let index = 0; index < 8400; index++) {
        const [class_id, prob] = [...Array(15).keys()]
            .map(col => [col, output[8400 * (col + 4) + index]])
            .reduce((accum, item) => item[1] > accum[1] ? item : accum, [0, 0]);

        if (prob < 0.5) {
            continue;
        }
        const label = yolo_classes[class_id];
        console.log("label", label)
        const xc = output[index];
        const yc = output[8400 + index];
        const w = output[2 * 8400 + index];
        const h = output[3 * 8400 + index];
        const x1 = (xc - w / 2) / 640 * img_width;
        const y1 = (yc - h / 2) / 640 * img_height;
        const x2 = (xc + w / 2) / 640 * img_width;
        const y2 = (yc + h / 2) / 640 * img_height;
        boxes.push([x1, y1, x2, y2, label, prob]);
    }

    boxes = boxes.sort((box1, box2) => box2[5] - box1[5]);
    console.log("boxes", boxes)
    const result = [];
    while (boxes.length > 0) {
        result.push(boxes[0]);
        boxes = boxes.filter(box => iou(boxes[0], box) < 0.7);
    }
    console.log("result", result)
    return result;
}


function iou(box1, box2) {
    return intersection(box1, box2) / union(box1, box2);
}

function union(box1, box2) {
    const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
    const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
    const box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1);
    const box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1);
    return box1_area + box2_area - intersection(box1, box2);
}

function intersection(box1, box2) {
    const [box1_x1, box1_y1, box1_x2, box1_y2] = box1;
    const [box2_x1, box2_y1, box2_x2, box2_y2] = box2;
    const x1 = Math.max(box1_x1, box2_x1);
    const y1 = Math.max(box1_y1, box2_y1);
    const x2 = Math.min(box1_x2, box2_x2);
    const y2 = Math.min(box1_y2, box2_y2);
    return (x2 - x1) * (y2 - y1);
}

async function predictImageSegmentation(model, image) {
    try {
        const [input, img_width, img_height] = await preprocessingImage(image);
        const rawOutput = await run_model(model, input);
        return process_output(rawOutput, img_width, img_height,);
    } catch (error) {
        console.error('Error processing image:', error);
        throw error;
    }
}

module.exports = {
    predictImageSegmentation,
};
