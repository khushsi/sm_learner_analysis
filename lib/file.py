__author__ = 'khushsi'

import os
import pickle


def chmod(cur_dir):
    print("\n===== Doing chmod  =====")
    ori_dir = os.path.dirname(os.path.realpath(__file__))

    command = "cd " + cur_dir
    print("command: ", command)
    os.chdir(cur_dir)

    command = "find . -type f -exec chmod 666 {} \;"
    print("command: ", command)
    os.system(command)

    command = "cd " + ori_dir
    print("command: ", command)
    os.chdir(ori_dir)


def create_file():
    ''' judge file or directory '''
    os.path.isfile('...')


def create_dir():
    dest_dir = '...'
    if not os.path.isdir(dest_dir):  # whether directory exists or not
        os.makedirs(dest_dir)  # make a tree of directory
        os.mkdir(dest_dir)  # make single layer directory


def copy_file(dest_dir):
    ''' copy file '''
    from shutil import copy
    file_path = '...'
    copy(file_path, dest_dir)


def get_current_path():
    script_path = os.path.dirname(os.path.realpath(__file__))
    ''' alternative way, get current executable directory '''
    encoding = os.sys.getfilesystemencoding()
    print os.path.dirname(unicode(__file__, encoding))


def change_filename(indir):
    import os
    print os.listdir(indir)
    [os.rename(indir + f, indir + f.replace('-', '')) for f in os.listdir(indir) if f.endswith('.jpg')]


def list_files(main_dir):
    files = [f for f in os.listdir(main_dir + "Input/") if os.path.isfile(os.path.join(main_dir + "Input/", f))]
    files = [f for f in os.listdir(main_dir) if os.path.isfile(os.path.join(main_dir, f)) if
             (not "feedback" in f and not "final_page" in f)]


def read_into_dict(infile, convert_to_int=False, sep=',', with_header=False):
    a_dict = {}
    i = 0
    with open(infile, 'r') as f:
        for line in f:
            if i == 0 and with_header:
                i += 1
                continue
            (key, val) = line.strip().split(sep)
            if convert_to_int:
                a_dict[int(key)] = int(val)
            else:
                a_dict[key] = val
            i += 1
    return a_dict


def saveObject(obj, name):
    pickle.dump(obj, open(name, 'wb'))


def loadObject(name):
    return pickle.load(open(file, 'rb'))
