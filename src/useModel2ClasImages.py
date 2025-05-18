import os
import shutil
import numpy as np
import cv2

import onnxruntime as ort
# import openvino as ov
from ultralytics import YOLO

def onnx_inference(model_pth, image_pth):
    # model_pth = os.path.join(model_dir, "best.onnx")
    assert os.path.isfile(model_pth), "Model File Not Exist"
    # CUDAExecutionProvider
    session = ort.InferenceSession(model_pth, providers=["CPUExecutionProvider"])
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_width = input_shape[2]
    input_height = input_shape[3]

    img = cv2.imread(image_pth)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_width, input_height))
    img_data = img.astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2,0,1))  # HWC  CHW
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)

    outputs = session.run(None, {model_inputs[0].name: img_data})
    result = outputs[0].reshape(-1)
    print("onnx_inference: ", result, " cls: ", np.argmax(result))
    return np.argmax(result)

def openvino_inference(model_dir, image_pth):
    model_pth = os.path.join(model_dir, "best.onnx")
    assert os.path.isfile(model_pth), "Model File Not Exist"
    core = ov.Core()
    model = core.read_model(model_pth)

    img = cv2.imread(image_pth)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0
    input_tensor = np.expand_dims(img, 0)
    ppp = ov.preprocess.PrePostProcessor(model)
    ppp.input().tensor().set_shape(input_tensor.shape).set_element_type(ov.Type.f32).set_layout(ov.Layout("NHWC"))
    ppp.input().preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    ppp.output().tensor().set_element_type(ov.Type.f32)
    model = ppp.build()

    compiled_model = core.compile_model(model, "CPU")
    results = compiled_model.infer_new_request({0: input_tensor})
    predictions = next(iter(results.values()))
    probs = predictions.reshape(-1)
    print("openvino_inference: ", probs)
    return

def ultralytics_inference(model_dir, image_pth):
    model_pth = os.path.join(model_dir, "best.pt")
    assert os.path.isfile(model_pth), "Model File Not Exist"

    model = YOLO(model_pth)
    res = model.predict(image_pth,
                        imgsz=224,
                        # device='cpu',
                        )
    print("ultralytics_inference: ", res[0].probs.data.cpu().numpy(), " cls: ", np.argmax(res[0].probs.data.cpu().numpy()))
    return np.argmax(res[0].probs.data.cpu().numpy())


CLS = ["NG", "OK"]
if __name__ == "__main__":

    res = onnx_inference(r"E:/Pretrained_models/YOLOv11-cls/yolo11n-cls.onnx", r"E:\DataSets\imageNet\n01443537_goldfish.JPEG")

    exit()

    src_dir = r"E:\DataSets\iqc_cracks\new_250307"
    # src_dir = r"E:\DataSets\iqc_cracks\origin_total\NG"
    # src_dir = r"E:\DataSets\OCR_qualified\original_gen\OK"
    save_dir = src_dir + r"\test_result"
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    for i in CLS:
        os.makedirs(os.path.join(save_dir, i))

    # model_dir = r"D:\share_dir\iqc_crack\ultr_workdir\crack_cls\0110_yolo11s_sgd_lr00052\weights"
    model_dir = r"D:\share_dir\iqc_crack\ultr_workdir\crack_cls0308\yolo11s_sgd_lr00052\weights"
    # image_pth = r"E:\DataSets\iqc_cracks\origin_total\OK\20241101000352112.jpg"
    # image_pth = r"E:\DataSets\iqc_cracks\origin_total\OK\20241027175758270.jpg"
    # image_pth = r"D:\share_dir\iqc_crack\ultr_workdir\crack_cls\0110_yolo11s_sgd_lr00052\weights\ng\20241101091439876.jpg"

    # img_dir = r"E:\DataSets\iqc_cracks\250307"
    for i, file in enumerate(os.listdir(src_dir)):
        if not file.endswith(".jpg"): continue
        print(f"{i}  ----  {file}")
        image_pth = os.path.join(src_dir, file)
        cls_id = onnx_inference(model_dir, image_pth)
        # openvino_inference(model_dir, image_pth)
        # cls_id = ultralytics_inference(model_dir, image_pth)

        shutil.copyfile(os.path.join(src_dir, file), os.path.join(save_dir, CLS[cls_id], file))

        print()
        print()

    exit(0)
    print("done")
