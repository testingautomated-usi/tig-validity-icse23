from os.path import exists, join
from os import makedirs
import numpy as np

def resize_image(img, size):
    from torchvision import transforms
    p = transforms.Resize(size)
    return p(img)

class Folder:
    IMG_COUNT = 0
    run_id = str(0)
    DST = "runs/run_" + run_id
    if not exists(DST):
        makedirs(DST)
    DST_UPTD = join(DST, "unperturbed")
    if not exists(DST_UPTD):
        makedirs(DST_UPTD)
    DST_ADV = join(DST, "adversarial")
    if not exists(DST_ADV):
        makedirs(DST_ADV)

    @staticmethod
    def save_img(img, original_img, target_cls, size):
        img_name = "seed_" + str(Folder.IMG_COUNT) + "_label_" + str(target_cls)
        img_path = join(Folder.DST_UPTD, img_name)
        resized_image = resize_image(original_img, size).cpu().detach()
        np.save(img_path, resized_image)

        img_name = "seed_"+str(Folder.IMG_COUNT)+"_label_"+str(target_cls)
        img_path = join(Folder.DST_ADV, img_name)
        resized_image = resize_image(img, size).cpu().detach()
        np.save(img_path, resized_image)
        Folder.IMG_COUNT += 1
