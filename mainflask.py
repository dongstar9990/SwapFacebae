
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import argparse
from face_detection import select_face, select_all_faces
from face_swap import face_swap
import random
app = Flask(__name__)

@app.route('/home', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded files
        src_file = request.files['src']
        dst_file = request.files['dst']
        my_list=[src_file , dst_file]
        val=random.choice(my_list)
        print(val)
        # Save the uploaded files to disk
        src_path = 'imgs/src_img.jpg'
        # dst_path = 'dst/dst_img.jpg'
        val.save(src_path)
        # dst_file.save(dst_path)
        # Swap faces
        args = argparse.Namespace(src=src_path, dst='imgs/gaixinh.jpg', out='results/output.jpg', warp_2d=False, correct_color=False, no_debug_window=True)
        src_img = cv2.imread(args.src)
        dst_img = cv2.imread(args.dst)
        src_points, src_shape, src_face = select_face(src_img)
        dst_faceBoxes = select_all_faces(dst_img)
        if dst_faceBoxes is None:
            print('Detect 0 Face !!!')
            exit(-1)
        output = dst_img
        print('dst_faceBoxes',dst_faceBoxes)
        for k, dst_face in dst_faceBoxes.items():
            output = face_swap(src_face, dst_face["face"], src_points, dst_face["points"], dst_face["shape"], output, args)
        output_path = 'results/output.jpg'
        cv2.imwrite(output_path, output)

        # Return the output image
        return send_file(output_path, mimetype='image/jpeg')

    # Render the HTML template
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
