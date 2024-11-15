'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''

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
    data_mode = 'Vid4'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
    # Vid4: SR
    # REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
    #        blur (deblur-clean), blur_comp (deblur-compression).
    stage = 1  # 1 or 2, use two stage strategy for REDS dataset.
    flip_test = False
    ############################################################################
    #### model
    if data_mode == 'Vid4':
        if stage == 1:
            model_path = '../experiments/pretrained_models/latest_G.pth'  #EDVR_Vimeo90K_SR_L.pth
        else:
            raise ValueError('Vid4 does not support stage 2.')
    elif data_mode == 'sharp_bicubic':
        if stage == 1:
            model_path = '../experiments/pretrained_models/EDVR_REDS_SR_L.pth'
        else:
            model_path = '../experiments/pretrained_models/EDVR_REDS_SR_Stage2.pth'
    elif data_mode == 'blur_bicubic':
        if stage == 1:
            model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_L.pth'
        else:
            model_path = '../experiments/pretrained_models/EDVR_REDS_SRblur_Stage2.pth'
    elif data_mode == 'blur':
        if stage == 1:
            model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_L.pth'
        else:
            model_path = '../experiments/pretrained_models/EDVR_REDS_deblur_Stage2.pth'
    elif data_mode == 'blur_comp':
        if stage == 1:
            model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_L.pth'
        else:
            model_path = '../experiments/pretrained_models/EDVR_REDS_deblurcomp_Stage2.pth'
    else:
        raise NotImplementedError

    if data_mode == 'Vid4':
        N_in = 3  # use N_in images to restore one HR image
    else:
        N_in = 3

    predeblur, HR_in, w_TSA = False, True, False
    back_RBs = 2  #40
    if data_mode == 'blur_bicubic':
        predeblur = True
    if data_mode == 'blur' or data_mode == 'blur_comp':
        predeblur, HR_in = True, True
    if stage == 2:
        HR_in = True
        back_RBs = 2
    model = MFDIN_arch.MFDIN_OLD2P(64, 4, 5, back_RBs)  #128

    #### dataset
    if data_mode == 'Vid4':
        test_dataset_folder = '../datasets/yk_test/DI'
        GT_dataset_folder = '../datasets/yk_test/GT_2P'
    else:
        if stage == 1:
            test_dataset_folder = '../datasets/REDS4/{}'.format(data_mode)
        else:
            test_dataset_folder = '../results/REDS-EDVR_REDS_SR_L_flipx4'
            print('You should modify the test_dataset_folder path for stage 2')
        GT_dataset_folder = '../datasets/REDS4/GT'

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode == 'Vid4' or data_mode == 'sharp_bicubic':
        padding = 'new_info'
    else:
        padding = 'replicate'
    save_imgs = True

    save_folder = '../results/{}'.format("yk_test2p")
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l, avg_ssim_center_l = [], [], [], []
    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    subfolder_GT_l = sorted(glob.glob(osp.join(GT_dataset_folder, '*')))
    # for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
            img_GT_l.append(data_util.read_img(None, img_GT_path))

        avg_ssim_center, avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            if flip_test:
                outputs = util.flipx4_forward(model, imgs_in)
            else:
                outputs = util.single_forward(model, imgs_in)
            outputs = outputs.squeeze(0)
            for i in range(2):
                gt_idx = 2 * img_idx + i
                output = util.tensor2img(outputs[i])
                if save_imgs:
                    nowname = str(gt_idx)
                    #nowname[-1] = chr(int(nowname[-1]) + i)
                    cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(nowname)), output)

                # calculate PSNR
                output = output / 255.
                #gt_idx = 2 * img_idx + i
                GT = np.copy(img_GT_l[gt_idx])
                # For REDS, evaluate on RGB channels; for Vid4, evaluate on the Y channel
                if data_mode == 'Vid4':  # bgr2y, [0, 1]
                    GT = data_util.bgr2ycbcr(GT, only_y=True)
                    output = data_util.bgr2ycbcr(output, only_y=True)

                output, GT = util.crop_border([output, GT], crop_border)
                crt_psnr = util.calculate_psnr(output * 255, GT * 255)
                crt_ssim = util.calculate_ssim(output * 255, GT * 255)
                logger.info('{:3d} - {:25} \tPSNR: {:.6f} dB'.format(gt_idx + 1, img_name, crt_psnr))

                if gt_idx >= border_frame and gt_idx < max_idx * 2 - border_frame:  # center frames
                    avg_psnr_center += crt_psnr
                    avg_ssim_center += crt_ssim
                    N_center += 1
                else:  # border frames
                    avg_psnr_border += crt_psnr
                    N_border += 1

        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_ssim_center = avg_ssim_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_ssim_center_l.append(avg_ssim_center)
        avg_psnr_border_l.append(avg_psnr_border)

        logger.info('Folder {} - Average PSNR: {:.6f} dB for {} frames; '
                    'Center PSNR: {:.6f} dB for {} frames; '
                    'Border PSNR: {:.6f} dB for {} frames.'.format(subfolder_name, avg_psnr,
                                                                   (N_center + N_border),
                                                                   avg_psnr_center, N_center,
                                                                   avg_psnr_border, N_border))

    logger.info('################ Tidy Outputs ################')
    for subfolder_name, psnr, psnr_center, psnr_border, ssim_center in zip(subfolder_name_l, avg_psnr_l,
                                                              avg_psnr_center_l, avg_psnr_border_l, avg_ssim_center_l):
        logger.info('Folder {} - Average PSNR: {:.6f} dB. '
                    'Center PSNR: {:.6f} dB. '
                    'Border PSNR: {:.6f} dB.'
                    'Center SSIM: {:.6f} dB. '.format(subfolder_name, psnr, psnr_center,
                                                     psnr_border, ssim_center))
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total Average PSNR: {:.6f} dB for {} clips. '
                'Center PSNR: {:.6f} dB. Border PSNR: {:.6f} dB. Center SSIM: {:.6f} dB'.format(
                    sum(avg_psnr_l) / len(avg_psnr_l), len(subfolder_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l),
                    sum(avg_ssim_center_l) / len(avg_ssim_center_l)))


if __name__ == '__main__':
    main()
