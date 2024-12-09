import os

def get_path_list(img_path, gt_path, fov_path, save_path):
    ## list names
    img_list, gt_list, fov_list = os.listdir(img_path), os.listdir(gt_path), os.listdir(fov_path)
    img_list.sort(), gt_list.sort(), fov_list.sort()
    ## save names
    with open(save_path, 'w') as f:
        for img, gt, fov in zip(img_list, gt_list, fov_list):
            f.write(img_path+'/'+img+' '+gt_path+'/'+gt+' '+fov_path+'/'+fov+'\n')

if __name__ == "__main__":
    get_path_list("./dataset/DRIVE/train/image", "./dataset/DRIVE/train/1st_manual", "./dataset/DRIVE/train/mask", "./dataset/DRIVE/train.txt")
    get_path_list("./dataset/DRIVE/test/image",  "./dataset/DRIVE/test/1st_manual",  "./dataset/DRIVE/test/mask",  "./dataset/DRIVE/test.txt" )