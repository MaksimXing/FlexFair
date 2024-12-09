import os

def get_path_list(img_path, gt_path, fov_path, save_train, save_test):
    ## list names
    img_list, gt_list, fov_list = os.listdir(img_path), os.listdir(gt_path), os.listdir(fov_path)
    img_list.sort(), gt_list.sort(), fov_list.sort()
    ## save test
    with open(save_test, 'w') as f:
        for img, gt, fov in zip(img_list[:5], gt_list[:5], fov_list[:5]):
            f.write(img_path+'/'+img+' '+gt_path+'/'+gt+' '+fov_path+'/'+fov+'\n')
    ## save train
    with open(save_train, 'w') as f:
        for img, gt, fov in zip(img_list[5:], gt_list[5:], fov_list[5:]):
            f.write(img_path+'/'+img+' '+gt_path+'/'+gt+' '+fov_path+'/'+fov+'\n')


if __name__ == "__main__":
    get_path_list("./dataset/STARE/image", "./dataset/STARE/1st_labels_ah", "./dataset/STARE/mask", "./dataset/STARE/train.txt", "./dataset/STARE/test.txt")
    
