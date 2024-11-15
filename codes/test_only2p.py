
import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.archs.MFDIN_arch as MFDIN_arch


def main():
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    ############################################################################
    #### model
    model_path = '../experiments/pretrained_models/MFDIN_old_2P.pth'  #EDVR_Vimeo90K_SR_L.pth

    N_in = 3  # use N_in images to restore one HR image
    back_RBs = 2
    model = MFDIN_arch.MFDIN_OLD2P(64, 4, 5, back_RBs)
    #### dataset
    test_dataset_folder = '../datasets/yk_test2p/DI'
    # temporal padding mode
    padding = 'new_info'
    save_imgs = True
    save_folder = '../results/{}'.format("yk_test2p")
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {}'.format(test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    subfolder_name_l = []
    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    # for each subfolder
    for subfolder in  subfolder_l:
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        #imgs_LQ = data_util.read_img_seq(subfolder)

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            #img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            img_path_l_in = [img_path_l[v] for v in select_idx]
            imgs_in = data_util.read_img_seq(img_path_l_in).unsqueeze(0).to(device)

            outputs = util.single_forward(model, imgs_in)
            outputs = outputs.squeeze(0)
            for i in range(2):
                gt_idx = 2 * img_idx + i
                output = util.tensor2img(outputs[i])
                if save_imgs:
                    nowname = '%06d' % gt_idx
                    cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(nowname)), output)

                logger.info('{:3d} - {:25}'.format(gt_idx + 1, nowname))

if __name__ == '__main__':
    main()
