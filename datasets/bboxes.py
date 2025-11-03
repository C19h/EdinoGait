import os
from tqdm import tqdm
from utils import gen_bboxes
from joblib import Parallel, delayed
root_dir = './db/HNU-Gait'

def main():
    tasks = []
    for c in os.listdir(root_dir):
        c_path = os.path.join(root_dir, c)
        for id_dir in os.listdir(c_path):
            id_path = os.path.join(c_path, id_dir)
            if os.path.isdir(id_path):
                for type_dir in os.listdir(id_path):
                    type_path = os.path.join(id_path, type_dir)
                    if os.path.isdir(type_path):
                        for view_dir in os.listdir(type_path):
                            view_path = os.path.join(type_path, view_dir)
                            file_path = os.path.join(view_path, 'bboxes.pkl')
                            if os.path.isfile(file_path):
                                continue
                            else:
                                tasks.append((view_path))


    Parallel(n_jobs=8)(
        delayed(gen_bboxes)(tasks[i]) for i in tqdm(range(len(tasks))))


def check():
    for c in os.listdir(root_dir):
        c_path = os.path.join(root_dir, c)
        for id_dir in os.listdir(c_path):
            id_path = os.path.join(c_path, id_dir)
            if os.path.isdir(id_path):
                for type_dir in os.listdir(id_path):
                    type_path = os.path.join(id_path, type_dir)
                    if os.path.isdir(type_path):
                        for view_dir in os.listdir(type_path):
                            file_path = os.path.join(type_path, view_dir, 'bboxes.pkl')
                            if os.path.isfile(file_path):
                                # os.remove(file_path)
                                # print(f"removed: {file_path}")
                                pass
                            else:
                                print(f"file: {file_path} no exist")


if __name__ == '__main__':
    main()
    # check()
    print('finish')
