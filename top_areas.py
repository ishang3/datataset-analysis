from shapely.geometry import Polygon
import os
import operator
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import cv2
import argparse
from helpers import plot
import numpy as np
import json


def create_histogram(dimensions,label):

    numberofvalues = len(dimensions)

    #calculates the width for each annotation across the labels
    width = [width for width,height in dimensions.values()] #95 percent

    width_zoom = sorted([width for width,height in dimensions.values()])[:int(.95*numberofvalues)]
    #calculates height of each annotations across the labels
    height = [height for width,height in dimensions.values()] #95 percent

    height_zoom = sorted([height for width, height in dimensions.
                        values()])[:int(.95 * numberofvalues)]
    #calculates aspect ratio of each annotation across the labels
    aspect_ratio = [width/height for width,height in dimensions.values()]

    aspect_ratio_zoom = sorted([width/height for width, height in dimensions.
                         values()])[:int(.95 * numberofvalues)]


    # calculates areas of each annotation across the labels
    areas = [(width*height)/100 for width,height in dimensions.values()]

    areas_zoom = sorted([(width*height)/100 for width,height in dimensions.
                   values()])[:int(.85*numberofvalues)]



    #bins = np.arange(start=min(areas),stop=300,step=30)
    # plot(width,label,'Width')
    # plot(height,label,'Height')
    # plot(aspect_ratio,label,'Aspect Ratio')
    plot(areas_zoom, label, 'Areas Zoom')



def crop_and_save(x,y,width,height,image,file_name,label):
    if not os.path.exists('output/cropped_anns_rgb'):
        os.mkdir('output/cropped_anns_rgb')
    if not os.path.exists('output/cropped_anns_gray'):
        os.mkdir('output/cropped_anns_gray')

    try:
        rgb_image = image[y:y+height,x:x+width]
        img_gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'output/{label}/cropped_anns_rgb/'+ str(number_of_annotations) + '-'
                    + file_name.split('/')[-1], rgb_image)
        cv2.imwrite(f'output/{label}/cropped_anns_gray/' + str(number_of_annotations) + '-'
                    + file_name.split('/')[-1], img_gray)
    except:
        print('DOES NOT WORK',file_name)


def driver(path,labels):
    total = {}

    global number_of_annotations
    number_of_annotations = 0
    for label in labels:
        dimensions = {} #refresh the list for each label
        for filename in os.listdir(f'{path}/'):
            file_end = filename.split('.')[-1]
            if file_end == 'txt':
                with open(f'{path}/'+filename) as fp:
                    line = fp.readline()
                    while line:
                        splitted = line.split(' ')
                        if splitted[0] == label:
                            number_of_annotations += 1

                            xmin = float(splitted[4])
                            ymin = float(splitted[5])
                            xmax = float(splitted[6])
                            ymax = float(splitted[7])

                            box = [[xmin,ymax],[xmax,ymax],[xmax,ymin],[xmin,ymin]]
                            width = xmax - xmin
                            height = ymax - ymin
                            annotation_filename = filename.split('.')[0] + '-' + str(number_of_annotations)
                            dimensions[annotation_filename] = (width,height)
                            box = Polygon(box)
                            total[filename] = box.area #this isnt used, but keeps a list of the files

                            # this will take the image and the ann coordinates
                            # and then save the cropped annotation in a separate folder
                            if crop:
                                img_path = path + filename.split('.')[0] + '.' + 'jpg'
                                image = cv2.imread(img_path)
                                crop_and_save(int(xmin),int(ymin),int(width),
                                  int(height),image,img_path,label)

                        #this will read the next line
                        line = fp.readline()

         #this will send the specific label's width and height
        print(label,'LABEL',len(dimensions),number_of_annotations)
        create_double(dimensions,label)
        #create_histogram(dimensions, label)




    #this sorts all the annotations by box area
    sorted_d = sorted(total.items(), key=operator.itemgetter(1))
    print(number_of_annotations)
    average_per_image =  number_of_annotations / len(total.items())

    with open('output/output.txt', 'a') as the_file:
        the_file.write(f'Average Number of Annotations per image {average_per_image}')
    print("average number of annotations",average_per_image)


def make_dirs(labels):
    for label in labels:
        if not os.path.exists(f'output/{label}'):
            os.mkdir(f'output/{label}')


if __name__ == '__main__':

    ## TODO

    ## width,height,area,average brightness,name of the file(primary key)



    #This script will return
    # 1) Histogram of the different range of pixel sizes
    # 2) Avg. Number of Annotations Per Image
    # 3) Crops each annotation in a new folder
    # 4) Interval Range for Histogram

    #Command Line Arguments
    #location to kitti file format

    if not os.path.exists('output'):
        os.mkdir('output')

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', '--dataset', help="Enter location of dataset; must be in kitti format")
    parser.add_argument('-crop', '--crop', help="To crop images annotations or not",default=False)
    parser.add_argument('-range', '--range', help="Interval for building histogram", default=60)
    parser.add_argument("--labels", nargs="+", default=["person"])

    args = parser.parse_args()

    global crop,interval
    crop = args.crop
    interval = args.range

    make_dirs(args.labels)

    args = parser.parse_args()
    driver(args.dataset,args.labels)
