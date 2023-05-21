import torch
from model import YOLOv3
from utils import non_max_suppression, cells_to_bboxes
import config
import cv2
import numpy as np
import time
import urllib.request

url = "http://127.0.0.1:5000/video_feed"


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    orig_im = img
    # dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_  # , orig_im, dim


def add_boxes_to_image(image, boxes):
    """
    Plots predicted bounding boxes on the image to CV2
    """
    class_labels = (
        config.COCO_LABELS if config.DATASET == "COCO" else config.PASCAL_CLASSES
    )
    im = np.array(image)
    height, width, _ = im.shape

    # Create a Rectangle patch
    for box in boxes:
        assert (
            len(box) == 6,
            "box should contain class pred, confidence, x, y, width, height",
        )

        class_pred = box[0]
        conf = box[1]
        box = box[2:]

        upper_left_x = (box[0] - box[2] / 2) * width
        upper_left_y = (box[1] - box[3] / 2) * height
        pt1 = (int(upper_left_x), int(upper_left_y))
        pt2 = (int(box[2] * width), int(box[3] * height))

        cv2.rectangle(
            image,
            pt1,
            pt2,
            [0, 0, 255],
            2,
        )

        label = f"{class_labels[int(class_pred)]}, {conf:.2}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

        end_point = pt1[0] + text_size[0] + 3, pt1[1] + text_size[1] + 4
        cv2.rectangle(
            image,
            pt1,
            end_point,
            [0, 0, 255],
            -1,
        )

        cv2.putText(
            image,
            label,
            (pt1[0], pt1[1] + text_size[1] + 4),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            [225, 255, 255],
            1,
        )


def main():
    # TODO: (aver) offload to gpu, otherwise very slow rendering
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    frames = 0
    start = time.time()
    # webcam = cv2.VideoCapture(0)  # try different number if not working
    # assert webcam.isOpened()

    stream = urllib.request.urlopen(url)
    bytes_buffer = b""

    while True:
        frames += 1

        # check, test_image = webcam.read()
        # if not check:
        #     break

        bytes_buffer += stream.read(1024)
        a = bytes_buffer.find(b"\xff\xd8")
        b = bytes_buffer.find(b"\xff\xd9")
        if a != -1 and b != -1:
            jpg = bytes_buffer[a : b + 2]
            bytes_buffer = bytes_buffer[b + 2 :]

            test_image = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
            )

            # Exit the loop if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if frames % 10 == 0:
                print("FPS: {:5.2f}".format(frames / (time.time() - start)))

            input_image = prep_image(test_image, config.IMAGE_SIZE)

            scaled_anchors = (
                torch.tensor(config.ANCHORS)
                * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
            ).to(config.DEVICE)

            # convert image to GPU
            input_image = input_image.to(config.DEVICE)

            with torch.no_grad():
                out = model(input_image)
                bboxes = [[] for _ in range(input_image.shape[0])]
                for i in range(3):
                    batch_size, _, S, _, _ = out[i].shape
                    anchor = scaled_anchors[i]
                    boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
                    for idx, (box) in enumerate(boxes_scale_i):
                        bboxes[idx] += box

            for i in range(batch_size):
                nms_boxes = non_max_suppression(
                    bboxes[i],
                    iou_threshold=0.6,
                    threshold=0.5,
                    box_format="midpoint",
                )
            # for i in range(batch_size):
            add_boxes_to_image(test_image, nms_boxes)

            cv2.imshow("final output", test_image)

    # webcam.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
