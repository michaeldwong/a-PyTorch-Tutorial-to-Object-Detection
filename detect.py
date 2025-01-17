import argparse
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
if not torch.cuda.is_available():
    checkpoint = torch.load(checkpoint,  map_location=torch.device('cpu'))
else:
    checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def detect(original_image, min_score, max_overlap, top_k, pid, frame_id=-1, suppress=None, draw_img=False):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if frame_id == -1:
        csv_file = f'{pid}-predictions.csv'
    else:
        csv_file = f'{pid}-{frame_id}-predictions.csv'
    if det_labels == ['background']:
        with open(csv_file, 'w') as f:
            f.write(f'left,top,right,bottom,class,confidence\n')
        # Just return original image
        return original_image

    if draw_img:
        # Annotate
        annotated_image = original_image
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.load_default()

        # Suppress specific classes, if needed
        
        for i in range(det_boxes.size(0)):
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue

            # Boxes
            box_location = det_boxes[i].tolist()
            draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
            # Text
            text_size = font.getsize(det_labels[i].upper())
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
            draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
            draw.text(xy=text_location, text=f'{det_labels[i].upper()} {det_scores[0][i].item() * 100}', fill='white',
                      font=font)
        del draw


    with open(csv_file, 'w') as f:
        f.write(f'left,top,right,bottom,class,confidence\n')
        for i in range(det_boxes.size(0)):
            if suppress is not None:
                if det_labels[i] in suppress:
                    continue
            print(det_labels[i])
            print(det_scores[0][i].item())
            # Boxes
            box_location = det_boxes[i].tolist()
            for j in range(len(box_location)):
                box_location[j] = int(max(box_location[j], 0))
            f.write(f'{int(box_location[0])},{int(box_location[1])},{int(box_location[2])},{int(box_location[3])},{det_labels[i]},{det_scores[0][i].item() * 100}\n')

    return annotated_image


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
    )
    ap.add_argument(
        "--frames",
        nargs="+",
        help="Frame numbers",
    )

    ap.add_argument(
        "--pid",
        help="PID for controllable camera experiments",
        default="0"
    )

    ap.add_argument(
        "--draw",
        action="store_true",
        help="Output jpg with bounding boxes",
    )

    args = ap.parse_args()
    for i,img in enumerate(args.input):
        original_image = Image.open(img, mode='r')
        original_image = original_image.convert('RGB')
        if args.frames:
            output = detect(original_image, min_score=0.4, max_overlap=0.5, top_k=200, pid=args.pid, frame_id=args.frames[i], draw_img=args.draw)
        else:
            output = detect(original_image, min_score=0.4, max_overlap=0.5, top_k=200, pid=args.pid, draw_img=args.draw)
        if args.draw:
            output.save(f'predictions.jpg')

