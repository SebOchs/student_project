import os
import xml.etree.ElementTree as et
import random

random.seed(1)


def split_xml(folder_path, new_folder_path):
    test = []
    for files in os.listdir(folder_path):
        if files.endswith('.xml'):
            tree = et.parse(folder_path + '/' + files)
            root = tree.getroot()

            # find student answers and split them
            stud_ans = root.find('studentAnswers')
            id_s = [x.attrib['id'] for x in stud_ans]
            random.shuffle(id_s)
            file_1 = [x for x in stud_ans if x.attrib['id'] in id_s[:14]]
            file_2 = [x for x in stud_ans if x.attrib['id'] in id_s[14:]]
            os.makedirs(new_folder_path + '/unseen_answers/', exist_ok=True)
            os.makedirs(new_folder_path + '/training/', exist_ok=True)
            # remove and replace for test set
            root.remove(root.find('studentAnswers'))
            child = et.Element('studentAnswers')
            for x in file_1:
                child.append(x)
            root.append(child)
            et.ElementTree(root).write(new_folder_path + '/unseen_answers/' + files)
            # remove and replace for train set
            root.remove(root.find('studentAnswers'))
            child = et.Element('studentAnswers')
            for x in file_2:
                child.append(x)
            root.append(child)
            et.ElementTree(root).write(new_folder_path + '/training/' + files)



split_xml('datasets/raw/kn1', 'datasets/raw/kn1')