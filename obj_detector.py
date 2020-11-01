import os as os
import sys as sys
import glob as glob

import dlib
from skimage import io
from skimage.draw import polygon_perimeter
import traceback


## Train the object detector based on the training data
if len(sys.argv) != 2:
    print(
        "Give the path to the image example e.g. testNutella1 directory as the argument to this "
        "program. For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./train_object_detector.py testNutella1")
    exit()
detector_folder = sys.argv[1]


## setting some options for detector such as symmetry
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True


## calling the svm model, training and testing data
training_xml = os.path.join(detector_folder,"train.xml")
testing_xml = os.path.join(detector_folder,"test.xml")
detector_svm = os.path.join(detector_folder,"detector.svm")

detector_svm

# This function does the actual training.  It will save the final detector to
# detector.svm.  The input is an XML file that lists the images in the training
# dataset and also contains the positions of the object boxes.  To create your
# own XML files you can use the imglab tool which can be found in the
# tools/imglab folder.  It is a simple graphical tool for labeling objects in
# images with boxes.  To see how to use it read the tools/imglab/README.txt
# file.  But for this example, we just use the training.xml file in the test folder.
if not os.path.exists(detector_svm):
    dlib.train_simple_object_detector(training_xml_path, detector_svm, options)


# Testing the object detector on the training and testing data.
print("Training accuracy")
print(dlib.test_simple_object_detector(training_xml,detector_svm)))
print("Testing accuracy")
print(dlib.train_simple_object_detector(testing_xml,detector_svm)))



## using the detector in the Nutella product
detector = dlib.simple_object_detector(detector_svm)
win_det = dlib.image_window()
win_det.set_image(detector)

## Now let's run the detector over the images in the test folder and display the
# results.
print("Showing detections on the images in the test folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(detector_folder, "test/*.jpg")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    dets = detector(img)
    print("Number of objects detected: {}".format(len(dets)))
    bOverLays = False
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        rr,cc = polygon_perimeter([d.top(), d.top(), d.bottom(), d.bottom()],
                             [d.right(), d.left(), d.left(), d.right()])
        try:
            img[rr, cc] = (255, 0, 0)
            if bOverLays == False:
                bOverLays = True
        except:
            traceback.print_exc()
    # Save the image detections to a file for future review.
    if bOverLays == True:
        io.imsave(f.replace("test/","output/"), img)
    win.clear_overlay()
    win.set_image(dlib.resize_image(img, 500, 500))
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()
    