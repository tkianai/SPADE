
import os
import cv2
import json
from flask import Flask, request, send_file
from datetime import timedelta

app = Flask(__name__)

app.send_file_max_age_default = timedelta(seconds=1)
app.config['JSON_AS_ASCII'] = False

ALLOWED_EXTENSIONS = set(
    ['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG', 'BMP'])


def allowed_file(filename):
    return '.' in filename and filename.split('.')[-1] in ALLOWED_EXTENSIONS

# ---------------------------------- #
import torch
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from PIL import Image
from data.base_dataset import get_params, get_transform
import util.util as util

opt = TestOptions().parse()
model = Pix2PixModel(opt)
model.eval()
print("model initialized!")
# ---------------------------------- #

def generate_image(input_path, output_path):
    # read input label file
    label = Image.open(input_path)
    params = get_params(opt, label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0
    label_tensor[label_tensor == 255] = opt.label_nc

    label_tensor = label_tensor.unsqueeze(0)

    input_dict = {
        'label': label_tensor,
        'instance': 0,
        'image': None,
        'path': input_path,
    }

    input_dict['label'] = input_dict['label'].long()
    input_dict['label'] = input_dict['label'].cuda()

    label_map = input_dict['label']
    bs, _, h, w = label_map.size()
    nc = opt.label_nc + 1 if opt.contain_dontcare_label else opt.label_nc
    input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    with torch.no_grad():
        fake_image = model.netG(input_semantics, z=None)

    fake_image = fake_image[0]
    fake_image = util.tensor2im(fake_image)
    util.save_image(fake_image, output_path, create_dir=True)

@app.route('/upload', methods=['POST'])
def index():

    content = request.files['file']
    if not (content and allowed_file(content.filename)):
        abort(400)

    # save image to fileSys.
    input_path = 'assets/' + content.filename
    output_path = 'assets/res_' + content.filename

    print('Saving to filepath ' + input_path)
    content.save(input_path)

    id_ = request.form['id']
    print("The request id is [{}]!".format(id_))

    generate_image(input_path, output_path)

    mimetype_str = 'image/' + output_path.split('.')[-1]
    return send_file(output_path, mimetype=mimetype_str)


if __name__ == '__main__':
    # app.debug = True
    app.run(host='10.141.8.84', port=8987, debug=False)
