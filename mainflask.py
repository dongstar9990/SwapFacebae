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
        # my_list=[src_file , dst_file]
        # val=random.choice(my_list)
        # print(val)
        # Save the uploaded files to disk
        src_path = 'imgs/src_img1.jpg'
        dst_path = 'imgs/src_img2.jpg'
        # val.save(src_path)

        src_file.save(src_path)
        dst_file.save(dst_path)

        # open image
        img = Image.open("imgs/couple.jpg")
        # lấy kích thước ảnh
        width, height = img.size

        # cắt lấy nửa ảnh đầu trên
        img_cropped1 = img.crop((0, 0, width//2 -40, height))
        # lưu ảnh đã cắt
        img_cropped1.save("imgs/img_1.jpg")
        # cắt lấy nửa ảnh đầu trên
        img_cropped2 = img.crop((width//2-40, 0, width, height))
        # lưu ảnh đã cắt
        img_cropped2.save("imgs/img_2.jpg")

        # Swap faces
        args = argparse.Namespace(src=src_path, dst='imgs/img_1.jpg', out='results/output1.jpg', warp_2d=False, correct_color=False, no_debug_window=True)
        src_img = cv2.imread(args.src)
        dst_img = cv2.imread(args.dst)
        src_points, src_shape, src_face = select_face(src_img)
        dst_faceBoxes = select_all_faces(dst_img)


        args1 = argparse.Namespace(src=dst_path, dst='imgs/img_2.jpg', out='results/output2.jpg', warp_2d=False, correct_color=False, no_debug_window=True)
        src_img2 = cv2.imread(args1.src)
        dst_img2 = cv2.imread(args1.dst)
        src_points2, src_shape2, src_face2 = select_face(src_img2)
        dst_faceBoxes2 = select_all_faces(dst_img2)

        if dst_faceBoxes is None:
            print('Detect 0 Face !!!')
            exit(-1)
        output = dst_img
        # print('dst_faceBoxes',dst_faceBoxes)

        if dst_faceBoxes2 is None:
            print('Detect 0 Face !!!')
            exit(-1)
        output2 = dst_img2

        for k, dst_face in dst_faceBoxes.items():
            output = face_swap(src_face, dst_face["face"], src_points, dst_face["points"], dst_face["shape"], output, args)
        output_path = 'results/output1.jpg'
        cv2.imwrite(output_path, output)


        for k, dst_face2 in dst_faceBoxes2.items():
                output2 = face_swap(src_face2, dst_face2["face"], src_points2, dst_face2["points"], dst_face2["shape"], output2, args1)
        output_path2 = 'results/output2.jpg'
        cv2.imwrite(output_path2, output2)
        # print("thanh cong ")

        image_1 = cv2.imread('results/output1.jpg')
        image_2 = cv2.imread('results/output2.jpg')

        # ghép hai ảnh lại với nhau theo chiều ngang
        combined_img = cv2.hconcat([image_1, image_2])
        result_img='results/output.jpg'
        # hiển thị ảnh đã ghép
        # cv2.imshow('Combined Image', combined_img)

        cv2.imwrite(result_img, combined_img)
        # Return the output image
        return send_file(result_img, mimetype='image/jpeg')

    # Render the HTML template
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')