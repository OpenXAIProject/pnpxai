import os

from PIL import Image, ImageOps
import matplotlib.colors as mcolors
import timm
from .model_normalization import ImageNetWrapper
from .load_smoothed_model import load_smoothed_model
from .load_trained_model import load_model as load_model_ratio

from .config import ask_overwrite_folder
import shutil
import torch.distributed as dist
import torch
import glob
import matplotlib.pyplot as plt
from .temperature_wrapper import TemperatureWrapper, get_temperature_folder
from utils_svces.config import image_loader, pretty, get_NVAE_MSE, get_NVAE_class_model, tensor_loader, models_dict, \
    Evaluator_model_names_cifar10, Evaluator_model_names_imagenet1000, descr_args_generate, descr_args_rst_stab, \
    loader_all, temperature_scaling_dl_dict, Evaluator_model_names_funduskaggle
from .GAN_metric import compute_score_raw
from .config import Evaluator_FID_base_path #Evaluator_FID_dict,
import pickle
import json
import numpy as np
import torchvision.transforms as transforms
from .config import FIDDataset, Evaluator_model_names_dict
from tqdm import tqdm
from robustbench import load_model as load_model_benchmark

import robust_finetuning.data as data_rf
import robust_finetuning.eval as utils_eval

from semisup_adv.smoothing import Smooth


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)




c = mcolors.ColorConverter().to_rgb

class Evaluator(object):
    def __init__(self, args, config, kwargs, dataloader):
        self.args = args
        self.config = config
        self.kwargs = kwargs
        self.dataloader = dataloader

    def load_model(self, model_id, return_preloaded_models=False):

        dataset = self.config.data.dataset_for_scorenet
        bs = self.config.sampling.batch_size
        folder = self.config.sampling.model_description.folder
        model_descr_args = {}
        folders = {}
        model_loaders = {}

        if dataset.lower() in ['cifar10', 'tinyimages']:
            type_ = Evaluator_model_names_cifar10[model_id]
        elif dataset.lower() in ['imagenet1000']:
            type_ = Evaluator_model_names_imagenet1000[model_id]
        elif dataset.lower() in ['funduskaggle']:
            type_ = Evaluator_model_names_funduskaggle[model_id]
        else:
            raise ValueError('Not implemented!')
        device = self.args.device

        if 'nonrobust' not in type_:
            descr_args = vars(self.config.sampling.model_description).copy()
            descr_args['device'] = self.args.device
            descr_args['dataset'] = dataset
            descr_args['type'] = type_

            if self.args.model_epoch_num is not None:
                descr_args['checkpoint'] = self.args.model_epoch_num

            if 'benchmark' in type_:
                # Overwrite ratio loader with one of the benchmarks loaders from https://github.com/RobustBench/robustbench or from
                # https://github.com/yaircarmon/semisup-adv
                model_name = type_.split('-')[1]
                is_experimental = 'improved' in model_name or 'experimental' in model_name
                is_Madry = 'Madry' in model_name
                is_Microsoft = 'Microsoft' in model_name
                is_Anon1s_small_radius = 'Anon1small_radius' in model_name
                is_MaxNets = 'Max:' in model_name
                is_Anon1_finetuning = 'Anon1:finetuning_experimental' in model_name
                is_SAM = 'SAM_experimental' in model_name

                if descr_args['dataset'].lower() in ['cifar10', 'tinyimages', 'imagenet1000', 'funduskaggle']:
                    # model_name in Gowal2020Uncovering_extra, Gowal2020Uncovering, Wu2020Adversarial
                    # add randomized smoothing!
                    if not is_experimental and len(type_.split('-')) == 3:
                        threat_model = type_.split('-')[2]
                    else:
                        threat_model = None

                    descr_args = descr_args_generate(threat_model=threat_model,
                                                     is_experimental=is_experimental,
                                                     model_name=model_name,
                                                     project_folder=self.args.project_folder)
                    if is_Madry:
                        arguments = model_name.split('_')
                        descr_args['norm'] = arguments[1]
                        assert arguments[2] == 'improved' or arguments[2] == 'experimental', \
                            'not a correct type of Madry model (only "improved" OR "experimental" are allowed)!'
                        descr_args['improved'] = arguments[2] == 'improved'
                        descr_args['num_pretrained_epochs'] = None if len(arguments) < 5 else arguments[4]
                        if '_eps_' in model_name:
                            assert len(arguments) == 5, 'Broken filename!'
                            descr_args['epsilon_finetuned'] = arguments[4]
                    elif is_Microsoft:
                        arguments = model_name.split(',')
                        descr_args['model_arch'] = arguments[0].split('Microsoft')[1]
                        descr_args['norm'] = arguments[2]
                        descr_args['epsilon'] = arguments[4]
                    elif is_Anon1s_small_radius:
                        descr_args['eps'] = model_name.split(':')[0]
                    elif is_MaxNets:
                        arguments = type_.split(',')
                        descr_args['dataset_name'] = dataset.lower()
                        descr_args['arch'] = arguments[1]
                        descr_args['model_name_id'] = arguments[2]
                        descr_args['num_classes'] = len(self.config.data.class_labels)
                        descr_args['img_size'] = self.config.data.image_size
                    elif is_Anon1_finetuning:
                        arguments = type_.split(',')
                        # Currently is only for 224x224 models!
                        descr_args['dataset_name'] = dataset.lower()
                        descr_args['arch'] = arguments[1].lower()
                        descr_args['model_name_id'] = arguments[2]
                        descr_args['num_classes'] = len(self.config.data.class_labels)

                    type_ = '-'.join(type_.split('-')[1:])

                    load_model_final = load_model_benchmark if not is_experimental else models_dict[
                        'Madry' if is_Madry else
                        'Anon1_small_radius_experimental' if is_Anon1s_small_radius else
                        'Microsoft' if is_Microsoft else
                        'Max' if is_MaxNets else
                        'Anon1:finetuning' if is_Anon1_finetuning else
                        'SAM' if is_SAM else
                        model_name]
                else:
                    raise ValueError('Benchmark robust models are only available for CIFAR10!')

            elif type_ == 'rst_stab':
                descr_args = descr_args_rst_stab
                load_model_final = load_smoothed_model
            else:
                if '_feature_model' in type_:
                    print('Loading feature comparison model')
                    descr_args['model_params'] = ['return_feature_map', True]
                    descr_args['type'] = type_.split('_')[0]

                load_model_final = load_model_ratio

        else:
            # ToDo: improve to remore the 'improved' suffix
            model_name = type_.split('_')[0] if '_' in type_ else type_
            descr_args = descr_args_generate(is_experimental=True,
                                             pretrained=(model_name == 'ResNet50IN1000' or 'timm' in type_),
                                             project_folder=self.args.project_folder)
            if 'BiT' in model_name or 'ViT' in model_name or 'CLIP' in type_:
                descr_args['model_name'] = model_name
                descr_args['class_labels'] = self.config.data.class_labels
                descr_args['dataset'] = dataset.lower()
                load_model_final = models_dict[model_name]
            elif 'ViT' in model_name:
                descr_args['device'] = device
                load_model_final = models_dict[model_name]
            elif 'timm' in type_:
                # ImageNet models used, with respective normalization
                model_name = type_.split(',')[1]
                load_model_final = lambda **kwargs: ImageNetWrapper(timm.create_model(**kwargs))
                descr_args['model_name'] = model_name
            elif model_name == 'ResNet50IN1000':
                load_model_final = models_dict[model_name]
            else:
                raise ValueError('Model is not implemented.')

        print('device is', device)
        print('device_ids are', self.args.device_ids)
        print('img_size is', self.config.data.image_size)
        print('dataset is', dataset)
        is_CLIP_model = False

        load_model = lambda x, loader, type_, folder: TemperatureWrapper(
            loader_all(is_CLIP_model, device, loader, x, self.args.device_ids),
            T=TemperatureWrapper.compute_temperature(
                loader_all(is_CLIP_model, device, loader, x, self.args.device_ids),
                temperature_scaling_dl_dict,
                bs,
                device=device,
                type_=type_,
                folder=folder,
                dataset=dataset,
                img_size=self.config.data.image_size,
                project_folder=self.args.project_folder,
                data_folder=self.args.data_folder
            ),
            # randomized_smoothing=self.config.sampling.ratio_mode.randomized_smoothing is not None and self.config.sampling.ratio_mode.randomized_smoothing,
            # noise_sigma=self.config.sampling.ratio_mode.randomized_smoothing_sigma
        )


        model_loaders[type_] = load_model_final
        model_descr_args[type_] = descr_args
        folders[type_] = folder
        print('descr args', model_descr_args[type_])

        if return_preloaded_models:
            return type_, model_loaders[type_], model_descr_args[type_], folders[type_]
        else:
            return [load_model(model_descr_args[type_], model_loaders[type_], type_, folders[type_]) for type_ in model_descr_args.keys()][0]

    def non_GAN_metrics_calculation(self):
        from NVAE_utils.utils import get_arch_cells
        from NVAE_utils.model import AutoEncoder
        device = self.args.device
        images_subfolder = 'FID_non_GAN_metrics_stats'
        images_folder_full = os.path.join(self.config.evaluation.base_folder, images_subfolder)
        ask_overwrite_folder(images_folder_full, True)
        classifiers_dict = self.kwargs['classifier']
        # Loading NVAE model
        checkpoint_path = 'NVAE/NVAE/checkpoints/eval-007/checkpoint.pt'
        checkpoint_paths_class_evals_dict = {'0': 0, '1': 1, '2': 2, '008': 3, '4': 4,
                                             '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, }
        class_models = {}
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        args = checkpoint['args']
        size = 1
        rank = 0
        local_rank = 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '1234'+str(self.args.model_type)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
        arch_instance = get_arch_cells(args.arch_instance)
        NVAE_model = AutoEncoder(args, None, arch_instance)
        NVAE_model.load_state_dict(checkpoint['state_dict'], strict=False)
        NVAE_model = NVAE_model.cuda()

        for key, value in checkpoint_paths_class_evals_dict.items():
            temp_path = f'NVAE/NVAE/checkpoints/eval-{key}/checkpoint.pt'
            checkpoint = torch.load(temp_path, map_location='cpu')
            args = checkpoint['args']
            arch_instance = get_arch_cells(args.arch_instance)
            class_models[value] = AutoEncoder(args, None, arch_instance)
            class_models[value].load_state_dict(checkpoint['state_dict'], strict=False)
            class_models[value] = class_models[value].cuda()

        # we assume only three model types for now - robust, non-robust and RATIO(ResNet50)
        RATIO_type = 'ResNet50'
        for type_ in classifiers_dict.keys():
            if 'nonrobust' in type_:
                nonrobust_type = type_
            elif type_ != RATIO_type:
                robust_type = type_

        for type_ in classifiers_dict.keys():
            classifiers_dict[type_].to(device)
            classifiers_dict[type_].eval()
        evaluator_model_names = Evaluator_model_names_dict[self.config.data.dataset.lower()]

        model_name = evaluator_model_names[self.args.model_type]
        experiments_eval_dict = {model_name: ''} #Evaluator_FID_dict[model_name]}

        for model_name, model_dict in experiments_eval_dict.items():
            for prior_or_pgd, prior_or_pgd_dict in model_dict.items():
                for dataset_name, paths in prior_or_pgd_dict.items():
                    paths = [os.path.join(Evaluator_FID_base_path, path) for path in paths]
                    model_stats_path = os.path.join(images_folder_full,
                                                    f'model:{model_name},type:{prior_or_pgd},dataset:{dataset_name}.pickle')

                    print('Saving in path', model_stats_path)
                    if os.path.isfile(model_stats_path):
                        print('Statistic has been calculated already!')
                    else:
                        dict_temp = {  # 'start': float(comma_split[0].split('=')[1]),
                            # 'end': float(comma_split[1].split('_')[0].split('=')[1]),
                            # we assume only robust and non-robust models in dict for now
                            'paths': [],
                            'p_model_nonrobust': [],
                            'p_model_robust': [],
                            'p_RATIO': [],
                            'NVAE_MSE': [],
                            'NVAE_MSE_c': []
                        }
                        dataset = FIDDataset(roots=paths, transform=transforms.Compose([transforms.ToTensor()]))
                        dataloader = torch.utils.data.DataLoader(
                            dataset, batch_size=self.config.sampling.batch_size, num_workers=self.config.data.num_workers)

                        #paths = [pathlib.Path(os.path.join(Evaluator_FID_base_path, path_)) for path_ in paths]
                        print(paths)
                        #files = sorted([file for path_ in paths for file in path_.glob('*_last.{}'.format('png'))])
                        #print('Number of files', len(files))
                        #dicts_list = []
                        for imgs, _, filepaths in tqdm(dataloader):
                            imgs = imgs.to(self.args.device)
                            probs_dict_temp = {}
                            #img = image_loader(filepath, self.config.data.image_size)

                            for type_ in classifiers_dict.keys():
                                out_temp = classifiers_dict[type_](imgs)
                                probs_dict_temp[type_] = torch.softmax(out_temp, dim=1)

                            #split_ = str(filepath).split('/')
                            #comma_split = split_[-1].split(',')
                            #target_class_name = split_[-1].split(':')[2]
                            #class_id = self.config.data.class_labels.index(target_class_name)
                            dict_temp['paths'].append(filepaths)
                            dict_temp['p_model_nonrobust'].append(probs_dict_temp[nonrobust_type])
                            dict_temp['p_model_robust'].append(probs_dict_temp[robust_type])
                            dict_temp['p_RATIO'].append(probs_dict_temp[RATIO_type])
                            dict_temp['NVAE_MSE'].append(get_NVAE_MSE(imgs, NVAE_model, batch_size=self.config.sampling.batch_size))
                            dict_temp['NVAE_MSE_c'].append([get_NVAE_MSE(imgs, get_NVAE_class_model(class_models, class_id), batch_size=self.config.sampling.batch_size) for class_id in range(10)])



                        print(f'Statistic for model {model_name} with type {prior_or_pgd} on the dataset'
                              f' {dataset_name} has been calculated!')
                        model_stats_path = os.path.join(images_folder_full,
                                                f'model:{model_name},type:{prior_or_pgd},dataset:{dataset_name}.pickle')

                        print('Saving in path', model_stats_path)

                        with open(model_stats_path, 'wb+') as f:
                            pickle.dump(dict_temp, f)
                        print('Stats data saved!')
                        """
                        for key in dict_temp:
                            mean_val = np.mean([dict_[key] for dict_ in dicts_list])
                            std_val = np.std([dict_[key] for dict_ in dicts_list])
                            print(f'Mean {key} is {mean_val}, std of it is {std_val}')
                        """

    @staticmethod
    def GAN_metrics_calculation():
        for model_name, model_dict in Evaluator_FID_dict.items():
            for prior_or_pgd, prior_or_pgd_dict in model_dict.items():
                for dataset_name, paths in prior_or_pgd_dict.items():
                    paths = [os.path.join(Evaluator_FID_base_path, path) for path in paths]
                    score = compute_score_raw(paths) #, device=self.args.device)
                    print(f'Model {model_name} with type {prior_or_pgd} on the dataset'
                          f' {dataset_name} has following scores:')
                    print(score)

    @torch.no_grad()
    def clean_accuracy(self, compute_ECE=True, compute_robust=True, compute_noise=False):
        temperature_scaling_str = lambda use_temp: 'classifier' if use_temp else 'classifier_no_temperature'
        temperature_scaling_arr = [True] if compute_robust else [True, False]
        if compute_robust:
            compute_ECE = False

        print('Wm intensity is', self.args.wm_intensity)
        for temperature_scaling in temperature_scaling_arr:
            for classifier_name, classifier in self.kwargs[temperature_scaling_str(temperature_scaling)].items():
                logits = []
                labels = []
                ##all_confidences = []
                #all_images = []
                #fake_granny_smith = []
                pred_confidences = []
                num_correct = 0
                num_correct_smoothed = 0
                num_samples = 0
                classifier.to(self.args.device)
                classifier.eval()

                if compute_noise:
                    print('Smoothing rst')
                    num_classes = len(self.config.data.class_labels)
                    sigma = 0.25
                    N0 = 100
                    N = 10000
                    alpha = 0.001
                    batch = 1000
                    #print('before smoothed classifier')
                    smoothed_classifier = Smooth(classifier, num_classes, sigma)
                    #print('defined smoothed classifier')
                show_labels = False
                #a = torch.load('ACSM/slurm_start_files/exp/logits_evolution_FID_final/image_samples/BENCH_FAILURE_CIFAR10_FID_apgd_75_2.5_l2_rst_stab_second_seed_1234/image_grid_non_denoised260000_eps:2.5:2:2.5_seed:1234_lr:1.6e-05_ns:1_gradx:10000_prior_sigma:1_second_best_init_ce_loss_projection:l2_grad_norm:l2_wrong_class_rst_stab_False/sample_torch_batch_num=1.0,batch_size=800.pt')
                #print('prob is', classifier(a[19, :, :, -32:].unsqueeze(0).cuda()).softmax(1))
                #print('prob is', classifier(a[19, :, :, :32].unsqueeze(0).cuda()).softmax(1))

                #max_batches = 2
                #cur_batch = 0
                for out in tqdm(self.dataloader):
                    #if cur_batch >= max_batches:
                    #    break
                    #cur_batch += 1

                    if len(out) == 3:
                        x, y, _ = out
                    elif len(out) == 2:
                        x, y = out

                    x = x.to(device=self.args.device)
                    y = y.to(device=self.args.device)

                    #print('before logits')
                    logits_ = classifier(x)


                    confidences_ = logits_.softmax(dim=1)
                    if show_labels:
                        print('labels', y)
                        print('x', x)
                        print('logits', logits_)
                        print('confid-s', confidences_)
                        show_labels = False

                    logits.append(logits_.detach().cpu())
                    labels.append(y.cpu())
                    pred_confidence, predictions = confidences_.max(dim=1)
                    pred_confidences.append(pred_confidence)
                    num_correct += (predictions == y).sum()

                    ##all_confidences.append(confidences_)
                    #all_images.append(x)
                    #fake_granny_smith.append(num_samples+torch.where((predictions != y) & (predictions == 948))[0])
                    if compute_noise:
                        prediction_smoothed, pAbar, radius, counts = smoothed_classifier.certify(
                            x, N0, N, alpha, batch)
                        num_correct_smoothed += (prediction_smoothed == y).sum()
                    num_samples += predictions.size(0)


                    #torch.save(torch.cat(all_images, dim=0),
                    #           f'ACSM/notebooks/all_images_{classifier_name}_intensity_0_watermark_Matthias_30k_check_again.pt')
                    #torch.save(torch.cat(fake_granny_smith, dim=0), 'ACSM/notebooks/fake_granny_smith_ids_second_probable_30k.pt')
                #print('Fake granny smith ids are', torch.cat(fake_granny_smith, dim=0))
                ##torch.save(torch.cat(all_confidences, dim=0),
                ##           f'ACSM/notebooks/all_confidences_correct_big_models_{classifier_name}_intensity_{self.args.wm_intensity}_watermark:{self.args.watermark_type}_Matthias_scaled_50k.pt')
                print(
                    f'Classifier {classifier_name} has accuracy {float(num_correct) / float(num_samples) * 100:.2f},'
                    f' and mean confidence of the predicted class is {torch.cat(pred_confidences).mean()}'
                    f' on the dataset {self.config.data.dataset} '
                    f'and with temperature scaling {"ON" if temperature_scaling else "OFF"}.')

                if compute_noise:
                    print(f'Evaluation with noise: accuracy is {float(num_correct_smoothed) / float(num_samples) * 100:.2f}')

                if compute_ECE:
                    logits = torch.cat(logits, 0)
                    labels = torch.cat(labels, 0)
                    ece = TemperatureWrapper.get_ece_inner(logits, labels)[0]
                    print(f'ECE is {ece}')

                if compute_robust:
                    if self.config.data.dataset.lower() == 'cifar10':
                        # ToDo - make it more general, not only for CIFAR10
                        n_ex_final = 1000
                        data_dir = '/home/scratch/datasets/CIFAR10'
                        all_norms = ['L2', '1.5', 'L1']
                        all_epss = [0.5, 1.5, 12]
                        batch_size_eval = 1000
                        x, y = data_rf.load_cifar10(n_ex_final, data_dir=data_dir, device='cpu')
                    elif self.config.data.dataset.lower() == 'imagenet1000':
                        all_norms = ['L2', '1.5', 'L1']
                        all_epss = [2, 12.5, 255]
                        n_ex_final = 1000
                        batch_size_eval = 200
                        x, y = data_rf.load_imagenet1000(n_ex_final, device='cpu')
                    elif self.config.data.dataset.lower() == 'funduskaggle':
                        all_norms = ['L2', 'L2', 'L2', 'L2', 'L2', 'L2']
                        all_epss = [0.1, 0.15, 0.2, 0.25, 0.5, 0.75]
                        n_ex_final = 1000
                        batch_size_eval = 200
                        x, y = data_rf.load_funduskaggle(n_ex_final, self.config.data, True, self.args.data_folder,
                              self.args.project_folder, device='cpu')
                        print('shapes', x.shape, y.shape)
                    else:
                        raise ValueError('Dataset is not implemented.')

                    _, final_acc_dets = utils_eval.eval_norms(classifier, x, y,
                                                                             l_norms=all_norms,
                                                                             l_epss=all_epss,
                                                                             bs=batch_size_eval,
                                                                             log_path=None,
                                                                             n_cls=len(self.config.data.class_labels))
                    print('Robust acc. details:', final_acc_dets)

    @torch.no_grad()
    def calculate_starting_probs_per_class(self, correct_only=False):
        probs_per_class = {i: [] for i in range(len(self.config.data.class_labels))}
        statistics_dict = {
            'mean_predicted_correct_trainset': [],
            '50th_percentile_predicted_correct_trainset': [],
            }

        for classifier_name, classifier in self.kwargs['classifier'].items():
            pred_correct = []
            labels = []
            pred_confidences = []
            num_correct = 0
            num_samples = 0
            classifier.to(self.args.device)
            classifier.eval()

            show_labels = False
            for x, y in tqdm(self.dataloader):
                x = x.to(device=self.args.device)
                y = y.to(device=self.args.device)

                logits_ = classifier(x)

                confidences_ = logits_.softmax(dim=1)
                if show_labels:
                    print('labels', y)
                    print('x', x)
                    print('logits', logits_)
                    print('confid-s', confidences_)
                    show_labels = False

                labels.append(y.cpu().numpy())
                pred_confidence, predictions = confidences_.max(dim=1)
                pred_confidences.append(pred_confidence.cpu().numpy())
                #if correct_only:
                pred_correct.append((predictions == y).cpu().numpy())

            labels_arr = np.concatenate(labels)
            pred_confidences_arr = np.concatenate(pred_confidences)
            pred_correct_arr = np.concatenate(pred_correct) if correct_only else True
            #print('pred_correct_arr is', all(pred_correct_arr))

            statistics_dict['pred_confidences_arr'] = pred_confidences_arr.tolist()
            statistics_dict['pred_correct_arr'] = np.concatenate(pred_correct).tolist()

            for i, probs in probs_per_class.items():
                print(i, np.sum(labels_arr == i))
                statistics_dict['mean_predicted_correct_trainset'].append(
                    float(pred_confidences_arr[np.where((labels_arr == i) & pred_correct_arr)].mean())
                )
                statistics_dict['50th_percentile_predicted_correct_trainset'].append(
                    float(np.median(pred_confidences_arr[np.where((labels_arr == i) & pred_correct_arr)]))
                )

            print(
                f'Classifier {classifier_name} has the following statistics:')
            stats_filename = get_temperature_folder(
                self.config.data.dataset_for_scorenet,
                classifier_name,
                self.config.sampling.model_description.folder)
            with open(os.path.join(stats_filename, 'class_specific_probs.pickle'), 'wb+') as f:
                pickle.dump(statistics_dict, f)
            print(json.dumps(statistics_dict, indent=4))

    @torch.no_grad()
    def latex_benchmark(self, table_type='benchmark'):
        """

        Parameters
        ----------
        table_type - benchmark, ablation, apgd, pgd&prior, afw&apgd, inverse, seed
        """
        ##device = self.args.device
        use_tensors = True

        use_offsets = False #True
        offset = self.config.data.image_size#150
        offset_2 = 20 #20
        images_folder = table_type  # 'benchmark_folder'  #'inverse_smaller_radius_latex_images'
        images_folder_full = os.path.join(self.config.evaluation.base_folder, '..', images_folder)
        ask_overwrite_folder(images_folder_full, True)
        ##classifiers_dict = self.kwargs['classifier']
        # Loading NVAE model
        """
        checkpoint_path = 'NVAE/NVAE/checkpoints/eval-007/checkpoint.pt'
        checkpoint_paths_class_evals_dict = {'0': 0, '1': 1, '2': 2, '008': 3, '4': 4,
                                             '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, }
        class_models = {}
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        args = checkpoint['args']
        size = 1
        rank = 0
        local_rank = 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '1234'
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
        arch_instance = get_arch_cells(args.arch_instance)
        NVAE_model = AutoEncoder(args, None, arch_instance)
        NVAE_model.load_state_dict(checkpoint['state_dict'], strict=False)
        NVAE_model = NVAE_model.cuda()
        

        for key, value in checkpoint_paths_class_evals_dict.items():
            temp_path = f'NVAE/NVAE/checkpoints/eval-{key}/checkpoint.pt'
            checkpoint = torch.load(temp_path, map_location='cpu')
            args = checkpoint['args']
            arch_instance = get_arch_cells(args.arch_instance)
            class_models[value] = AutoEncoder(args, None, arch_instance)
            class_models[value].load_state_dict(checkpoint['state_dict'], strict=False)
            class_models[value] = class_models[value].cuda()
        """
        """
        # we assume only three model types for now - robust, non-robust and RATIO(ResNet50)
        RATIO_type = 'ResNet50'
        for type_ in classifiers_dict.keys():
            if 'nonrobust' in type_:
                nonrobust_type = type_
            elif type_ != RATIO_type:
                robust_type = type_

        for type_ in classifiers_dict.keys():
            classifiers_dict[type_].to(device)
            classifiers_dict[type_].eval()
        """
        if self.config.data.dataset == 'ImageNet1000':
            if table_type in ['ablation']:
                models_to_use = {i: self.load_model(i) for i in [0, 1, 3, 4, 5, 6, 7]}
                for i in models_to_use:
                    models_to_use[i].eval()

                subdir_prefixes_names_to_use = [##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                                        (['ResNet50IN1000'], 'ResNet50IN1000', 16, 60.3, models_to_use[6]),

                                        (['Madry_l2_experimental'], 'l2', 4.9, 22.8, models_to_use[0]),
                                        # ('TEST-correct_ResNet50-nonrobust-randomized-smoothing', 'NR-R50+RS'),
                                        #(['Madry_l2_improved_ep_1'], 'l2_ep1', 4.3, 23),
                                        # ('TEST-correct_RST_stab-randomized-smoothing', 'RST-stab+RS'),
                                        (['Madry_l2_improved_ep_3'], 'l2_ep3', 4.3, 23.1, models_to_use[3]),
                                        ##('Gowal2020Uncovering', 'GU', 8.24, 68.04),
                                        (['Madry_linf_experimental'], 'linf', 6.3, 34.9, models_to_use[1]),
                                        ##('Wu2020Adversarial', 'WuAdversarial', 7.96, 67.38),
                                        (['Madry_linf_improved_ep_1'], 'linf_ep1', 4.3, 23.3, models_to_use[5]),
                                        (['Madry_l2_improved_ep_1l1'], 'l2_ep1l1', 5.6, 28.5, models_to_use[7]),
                                       ]

            elif table_type == 'apgd':
                # ToDo: we assume that for l1 and l2 norms are of different sizes
                # and thus no explicit reference to the norm name is used

                norms_eps_arr = list(
                    zip(['l_2'] * 4 + ['l_1'] * 4, ['6.0', '8.0', '10.0', '12.0', '600.0', '800.0', '1000.0', '1200.0']))
                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    (['eps:' + eps, 'Madry_l2_improved_ep_3'], f'${norm_name}:{eps}$') for norm_name, eps in norms_eps_arr
                ]
            elif table_type == 'inverse':
                # ToDo: we assume that for l1 and l2 norms are of different sizes
                # and thus no explicit reference to the norm name is used
                norms_eps_arr = list(
                    zip(['inverse_75_l1'] * 4 + ['inverse_75_l2'] * 4, 2 * (['0.8'] + ['0.'+'9'*i for i in range(1, 4)])))
                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    ([type_inverse + '_' + threshold_name + '_'], '$l_1$' + ':$' + threshold_name + '$' if 'l1' in type_inverse else '$l_2$' + ':$' + threshold_name + '$' if 'l2' in type_inverse else '$l_{1.5}$' + ':$' + threshold_name + '$') for type_inverse, threshold_name in norms_eps_arr
                ]

            elif table_type == 'afw':
                # ToDo: we assume that for l1 and l2 norms are of different sizes
                # and thus no explicit reference to the norm name is used
                models_to_use = {i: self.load_model(i) for i in [3]}
                for i in models_to_use:
                    models_to_use[i].eval()

                """
                norms_eps_arr = list(
                    zip(['afw_75'] + ['apgd_75'],
                        [ '1.5', 'l2'],
                        ['Madry_l2_ep3'] * 2,
                        ['50', '12'],
                        [6, 4.3],
                        [35.7, 23.1]
                        ))
                """

                """
                norms_eps_arr = list(
                    zip(['apgd_75'] + ['afw_75'] + ['apgd_75'],
                        ['l1',  '1.5', 'l2'],
                        ['Madry_l2_ep3'] * 3,
                        ['400', '50', '12'],
                        [6, 4.1, 4.3],
                        [35.7, 22.7, 23.1]
                        ))
                """
                norms_eps_arr = list(
                    zip(['afw_75'] + ['apgd_75'],
                        ['l2', 'l2'],
                        ['Madry_l2_ep3'] * 2,
                        ['12', '12'],
                        [4.1, 4.3],
                        [22.7, 23.1]
                        ))
                """
                norms_eps_arr = list(
                    zip(['apgd_75'] + ['afw_75']*4 + ['apgd_75'],
                        ['l1', '1.5', '1.5', '1.5', 'l2', 'l2'],
                        ['Madry_l2_ep3']*6,
                        ['400', '40', '50', '60', '12', '12'],
                        [6, -1, -1, -1, 4.1, 4.3],
                        [35.7, -1, -1, -1, 22.7, 23.1]
                ))
                """


                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    ([type_afw + '_' + eps + '_' + norm_name + '_' + model_name] if 'afw' in type_afw else [type_afw + '_' + norm_name + '_' + eps + '_' + model_name],
                     '$l_{' + norm_name + '}:' + eps + '$' if norm_name != 'l2' else '$l_2$:'+eps,
                     fid_id,
                     fid_od, models_to_use[3]) for type_afw, norm_name, model_name, eps, fid_id, fid_od in norms_eps_arr
                ]

            elif table_type == 'diversity_ODI':
                type_to_name = {'ODI': 'ODI',
                                'iter_increase': 'iter_increase'}

                type_to_steps = {'ODI': '160__FAILURE',
                                 'iter_increase': '235_125_1.5'}
                norms_eps_arr = list(
                    zip(['ODI'] * 4 + ['iter_increase'] * 4, 2 * [str(i) + '000' for i in range(1, 5)]))
                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    ([seed, 'Madry_l2_ep3', type_to_steps[type_seed], '125_1.5'], f'{type_to_name[type_seed]}:${seed[0]}e4$') for type_seed, seed in
                    norms_eps_arr
                ]

        else:
            if table_type == 'benchmark':
                models_to_use = {i: self.load_model(i) for i in [0, 1, 4, 6, 8, 15, 21]}
                for i in models_to_use:
                    models_to_use[i].eval()

                subdir_prefixes_names_to_use = [
                    (['BiT-M-R50x1_CIFAR10_nonrobust'], 'BiT-M', 51, 83.9, models_to_use[1]),
                    # ('TEST-correct_ResNet50-nonrobust-randomized-smoothing', 'NR-R50+RS'),
                    (['rst_stab'], 'RST-stab', 5, 50.7, models_to_use[4]),
                    # ('TEST-correct_RST_stab-randomized-smoothing', 'RST-stab+RS'),
                    #(['ResNet50'], 'RATIO', 8.4, 28.2),
                    (['Augustin2020Adversarial_34_10_extra-L2'], 'RATIO', 8.4, 28.2, models_to_use[21]), # ToDo: change FID
                    ##('Gowal2020Uncovering', 'GU', 8.24, 68.04),
                    (['Gowal2020Uncovering_extra-L2'], 'GU-extra', 14.8, 53.9, models_to_use[0]),
                    (['Gowal2020Uncovering_improved'], 'GU-impr', 10, 41.8, models_to_use[6]),
                    ##('Wu2020Adversarial', 'WuAdversarial', 7.96, 67.38),
                    (['PAT_improved'], 'PAT', 11.7, 49, models_to_use[8]),
                    (['Hendrycks2020AugMix_ResNeXt-corruptions'], 'HenC', 32.1, 67.2, models_to_use[15])
                    ]
                """
                    (['0.02l2:Anon1_small_radius_experimental'], 'l2:0.02', 15, 49.4),
                    (['0.1l2:Anon1_small_radius_experimental'], 'l2:0.1', 14, 47.2),
                    (['0.25l2:Anon1_small_radius_experimental'], 'l2:0.25', 13.5, 44.7),
                    (['0.5l2:Anon1_small_radius_experimental'], 'l2:0.5', 11.5, 47.1),
                    (['0.75l2:Anon1_small_radius_experimental'], 'l2:0.75', 10.6, 49.6),
                    (['1l2:Anon1_small_radius_experimental'], 'l2:1', 8.7, 50.1),
                    (['8,255linf:Anon1_small_radius_experimental'], 'linf:8,255', 15.1, 60.7),
                    (['12l1:Anon1_small_radius_experimental'], 'l1:12', 14.3, 48.7),
                    ]
                """
                ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),

                ##('rlat-eps=0.05-augmix', 'RLAT', 53.28, 91.41),
                """
                (['0.02l2:Anon1_small_radius_experimental'], 'l2:0.02', 15, 49.4),
                (['0.1l2:Anon1_small_radius_experimental'], 'l2:0.1', 14, 47.2),
                (['0.25l2:Anon1_small_radius_experimental'], 'l2:0.25', 13.5, 44.7),
                (['0.5l2:Anon1_small_radius_experimental'], 'l2:0.5', 11.5, 47.1),
                (['0.75l2:Anon1_small_radius_experimental'], 'l2:0.75', 10.6, 49.6),
                (['1l2:Anon1_small_radius_experimental'], 'l2:1', 8.7, 50.1),
                (['8,255linf:Anon1_small_radius_experimental'], 'linf:8,255', 15.1, 60.7),
                (['12l1:Anon1_small_radius_experimental'], 'l1:12', 14.3, 48.7),
                """
            elif table_type == 'ablation':

                models_to_use = {i: self.load_model(i) for i in [6, 11, 12, 16, 17, 18, 19, 20]}
                for i in models_to_use:
                    models_to_use[i].eval()

                subdir_prefixes_names_to_use = [
                    (['Gowal2020Uncovering_improved'], 'GU-impr', 10, 41.8, models_to_use[6]),
                    #(['0.02l2:Anon1_small_radius_experimental'], 'l2:0.02', 15, 49.4, models_to_use[10]),
                    (['0.1l2:Anon1_small_radius_experimental'], 'l2:0.1', 14, 47.2, models_to_use[11]),
                    (['0.25l2:Anon1_small_radius_experimental'], 'l2:0.25', 13.5, 44.7, models_to_use[12]),
                    (['0.5l2:Anon1_small_radius_experimental'], 'l2:0.5', 11.5, 47.1, models_to_use[16]),
                    (['0.75l2:Anon1_small_radius_experimental'], 'l2:0.75', 10.6, 49.6, models_to_use[20]),
                    (['1l2:Anon1_small_radius_experimental'], 'l2:1', 8.7, 50.1, models_to_use[19]),
                    (['8,255linf:Anon1_small_radius_experimental'], 'linf:8,255', 15.1, 60.7, models_to_use[18]),
                    (['12l1:Anon1_small_radius_experimental'], 'l1:12', 14.3, 48.7, models_to_use[17]),
                    ]

                ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),

                ##('rlat-eps=0.05-augmix', 'RLAT', 53.28, 91.41),

            elif table_type == 'ood':
                pass
            elif table_type == 'apgd':
                # ToDo: we assume that for l1 and l2 norms are of different sizes
                # and thus no explicit reference to the norm name is used
                norms_eps_arr = list(zip(['l_2']*4 + ['l_1']*4, ['1.0', '1.5', '2.0', '2.5', '10.0', '15.0', '20.0', '25.0'])) # '40', '60', '80', '100']))
                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    (['eps:'+eps, 'Gowal2020Uncovering_improved'], f'${norm_name}:{eps}$') for norm_name, eps in norms_eps_arr
                ]
            elif table_type == 'afw':
                # ToDo: we assume that for l1 and l2 norms are of different sizes
                # and thus no explicit reference to the norm name is used
                """
                norms_eps_arr = list(
                    zip(
                        ['apgd_75'] + ['intersection'] * 6 + ['apgd_75'],
                        ['l1', '1.5', '1.5', '1.5', '1.5', '1.5', 'l2', 'l2'],
                        ['20', '5', '6', '7.5', '10', '12.5', '2.5', '2.5'],
                        [6.1, -1, 6.6, -1, -1, -1, 9.9, 10],
                        [52.7, -1, 43.4, -1, -1, -1, 41.2, 41.2]
                        )
                )
                """

                """
                norms_eps_arr = list(
                    zip(
                        ['apgd_75'] + ['intersection'] * 4 + ['apgd_75'],
                        ['l1', '1.5', '1.5', '1.5', 'l2', 'l2'],
                        ['20', '5', '6', '7', '2.5', '2.5'],
                        [6.1, -1, 6.6, -1, 9.9, 10],
                        [52.7, -1, 43.4, -1, 41.2, 41.2]
                    )
                )
                """
                norms_eps_arr = list(
                    zip(
                        ['intersection'] + ['apgd_75'],
                        ['l2', 'l2'],
                        ['2.5', '2.5'],
                        [9.9, 10],
                        [41.2, 41.2]
                    )
                )
                models_to_use = {6: self.load_model(6)}
                for i in models_to_use:
                    models_to_use[i].eval()
                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    ([type_afw + '_' + norm_name + '_' + eps + '_GU_impr'],
                     '$l_{' + norm_name + '}:' + eps + '$' if norm_name != 'l2' else '$l_2$:'+eps, fid_id, fid_od,
                     models_to_use[6]) for type_afw, norm_name, eps, fid_id, fid_od in norms_eps_arr
                ]
            elif table_type == 'inverse':
                # ToDo: we assume that for l1 and l2 norms are of different sizes
                # and thus no explicit reference to the norm name is used
                norms_eps_arr = list(
                    zip(['inverse_75_l1'] * 4 + ['inverse_75_l2'] * 4, 2 * ['0.'+'9'*i for i in range(2, 6)]))
                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    ([type_inverse + '_' + threshold_name + '_'], '$l_1$' + ':$' + threshold_name + '$' if 'l1' in type_inverse else '$l_2$' + ':$' + threshold_name + '$' if 'l2' in type_inverse else '$l_{1.5}$' + ':$' + threshold_name + '$') for type_inverse, threshold_name in norms_eps_arr
                ]
            elif table_type == 'diversity':
                # ToDo: we assume that for l1 and l2 norms are of different sizes
                # and thus no explicit reference to the norm name is used
                models_to_use = {6: self.load_model(6)}
                for i in models_to_use:
                    models_to_use[i].eval()
                type_to_name = {'apgd': 'APGD',
                                'prior_pgd': 'PALMC'}

                norms_eps_arr = list(
                    zip(['apgd'] * 4 + ['prior_pgd'] * 4, 2 * [str(i) + '000' for i in range(1, 5)]))
                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    ([type_seed, seed, 'GU_impr'], f'{type_to_name[type_seed]}:${seed[0]}e4$', models_to_use[6]) for type_seed, seed in norms_eps_arr
                ]
            elif table_type == 'diversity_ODI':
                type_to_name = {'ODI': 'ODI',
                                'iter_increase': 'iter_increase'}

                type_to_steps = {'ODI': '160_FAILURE',
                                 'iter_increase': '235_5_intersection_1.5_15'}
                norms_eps_arr = list(
                    zip(['ODI'] * 4 + ['iter_increase'] * 4, 2 * [str(i) + '000' for i in range(1, 5)]))
                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    ([type_seed, seed, 'GU_impr', type_to_steps[type_seed]], f'{type_to_name[type_seed]}:${seed[0]}e4$') for type_seed, seed in
                    norms_eps_arr
                ]

            elif table_type == 'pgd&prior':
                marker_to_label_dict = {
                                        'apgd_75_l2': '$l_2$-VCE,$\epsilon=2.5',
                                        #'apgd_75_l1': 'APGD,$l_1',
                                        #'nsigma_l1': 'PSGLA,$l_1',
                                        'nsigma_l2': '$l_2$-pVCE, $\epsilon=2.5'
                                        }
                norms_eps_arr = list(
                    zip(['apgd_75_', 'nsigma_'] * 4, ['2.5_l2', '2.5_l2', '5_l2', '5_l2']))
                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    ([prior_marker+eps, 'GU_impr'], f'{marker_to_label_dict[prior_marker+eps.split("_")[-1]]}$:${eps.split("_")[0]}$') for prior_marker, eps in norms_eps_arr
                ]
            elif table_type == 'prior':
                marker_to_label_dict = {
                    # 'apgd_75_l1': 'APGD,$l_1',
                    # 'nsigma_l1': 'PSGLA,$l_1',
                    'sigma:50seed:1234': '$l_2$-pVCE, $epsilon=2.5'
                }

                models_to_use = {i: self.load_model(i) for i in [0, 1, 4, 6, 8, 15, 21]}
                for i in models_to_use:
                    models_to_use[i].eval()

                norms_eps_arr = list(
                    zip(['sigma:50']*7, ['2.5:2:2.5_seed:1234']*7))

                model_name_loader = [
                    ('BiT-M-R50x1_CIFAR10_nonrobust', models_to_use[1]),
                    # ('TEST-correct_ResNet50-nonrobust-randomized-smoothing', 'NR-R50+RS'),
                    ('rst_stab', models_to_use[4]),
                    # ('TEST-correct_RST_stab-randomized-smoothing', 'RST-stab+RS'),
                    #(['ResNet50'], 'RATIO', 8.4, 28.2),
                    ('Augustin2020Adversarial_34_10_extra-L2', models_to_use[21]), # ToDo: change FID
                    ##('Gowal2020Uncovering', 'GU', 8.24, 68.04),
                    ('Gowal2020Uncovering_extra-L2', models_to_use[0]),
                    ('Gowal2020Uncovering_improved', models_to_use[6]),
                    ##('Wu2020Adversarial', 'WuAdversarial', 7.96, 67.38),
                    ('PAT_improved', models_to_use[8]),
                    ('Hendrycks2020AugMix_ResNeXt-corruptions', models_to_use[15])
                    ]

                subdir_prefixes_names_to_use = [  ##('ResNet50-nonrobust-correct', 'NR-R50', 25.34, 82.75),
                    ([model_name_loader[i][0], prior_marker, eps],
                     f'{model_name_loader[i][0][:7]},{marker_to_label_dict[prior_marker + eps.split("_")[-1]]}$:${eps.split("_")[0]}$', model_name_loader[i][1]) for
                    i, (prior_marker, eps) in enumerate(norms_eps_arr)
                ]

        images_info_dict = {j: {} for j in self.config.evaluation.ids}
        diff_plots_tabletypes = ['afw', 'pgd&prior', 'diversity_ODI', 'apgd', 'benchmark', 'ablation', 'prior']

        keys = []
        max_diff_pixels = 0
        min_diff_pixels = 1
        pixels = []

        for filepath in glob.glob(os.path.join(self.config.evaluation.base_folder,
                                               #'**',
                                               self.config.evaluation.pattern_folder,
                                               '**',
                                               '*.png'), recursive=True):
            current_class = list(filter(lambda class_num: f'/{class_num}_' in filepath, self.config.evaluation.ids))
            if len(current_class) == 1:
                ##probs_dict_temp = {}
                #img = image_loader(filepath, self.config.data.image_size)
                ##img = tensor_loader(filepath, self.config.data.image_size, current_class[0])
                ##for type_ in classifiers_dict.keys():
                ##    out_temp = classifiers_dict[type_](img)
                ##    probs_dict_temp[type_] = torch.softmax(out_temp, dim=1)

                split_ = filepath.split('/')
                index_to_search = -2 if table_type in ['apgd', 'benchmark', 'ablation', 'prior'] else -3 if table_type in ['pgd&prior', 'afw', 'inverse', 'diversity', 'diversity_ODI'] else -2
                # prefix, name, FID_CIFAR10, FID_TinyImages
                for kwargs in subdir_prefixes_names_to_use:
                    # if split_[-2].startswith(prefix + '_eps') or prefix in split_[-2]:

                    print('current class is', current_class, 'prefix is', kwargs[0], 'split is', split_[index_to_search])
                    if all(prefix_ in split_[index_to_search] for prefix_ in kwargs[0]):

                        prefix = kwargs[0][0]
                        name = kwargs[1]
                        model = kwargs[-1]
                        filename = prefix + split_[-1][:10].replace('$', '!')

                        print('filepath is',
                              '/'.join(filepath.split('/')[:-1]) + '/sample_torch_batch_num=1.0,batch_size=*.pt')
                        filepath_detected = glob.glob(
                            '/'.join(filepath.split('/')[:-1]) + '/sample_torch_batch_num=1.0,batch_size=*.pt')[0]
                        print('loading tensor of images from', filepath_detected)
                        imgs_tensor = torch.load(filepath_detected)

                        if prefix not in images_info_dict[current_class[0]]:
                            print('filepath is', '/'.join(filepath.split('/')[:-1])+'/sample_torch_batch_num=1.0,batch_size=*.pt')
                            filepath_detected = glob.glob('/'.join(filepath.split('/')[:-1])+'/sample_torch_batch_num=1.0,batch_size=*.pt')[0]
                            print('loading tensor of images from', filepath_detected)
                            images_info_dict[current_class[0]][prefix] = {}
                        if 'init_image_path' not in images_info_dict[current_class[0]]:
                            images_info_dict[current_class[0]]['init_image_path'] = ''
                            images_info_dict[current_class[0]]['correct_image_class'] = ''
                            images_info_dict[current_class[0]]['wrong_image_class'] = ''
                        if '_first.png' in split_[-1]:
                            if images_info_dict[current_class[0]]['init_image_path'] == '':
                                images_info_dict[current_class[0]]['init_image_path'] = filename+'.png'

                                if use_offsets:  #offset != self.config.data.image_size:
                                    img_pil = transforms.ToPILImage()(imgs_tensor[current_class[0] - 1, :, -offset:self.config.data.image_size, self.config.data.image_size-offset:self.config.data.image_size])
                                    if use_offsets:  # offset != self.config.data.image_size:
                                        zoom_in = img_pil.crop((100, 135, 150, 185)).resize((110, 110))
                                        zoom_in_with_border = ImageOps.expand(zoom_in, border=1, fill='red')
                                        img_pil.paste(zoom_in_with_border, (112, 0))
                                        img_pil.paste(
                                            ImageOps.expand(
                                                img_pil.crop((100, 135, 150, 185)), border=1, fill='red')
                                            , (99, 134)
                                        )
                                else:
                                    img_pil = transforms.ToPILImage()(imgs_tensor[current_class[0] - 1, :, :, :self.config.data.image_size])


                                img_pil.save(os.path.join(images_folder_full, 'init_' + filename + '.png'))
                                #shutil.copy(filepath, os.path.join(images_folder_full, 'init_' + filename + '.png'))
                        else:
                            correct_wrong = 'correct' if '_correct_' in split_[-2] else 'wrong'

                            if images_info_dict[current_class[0]][correct_wrong + '_image_class'] == '':
                                images_info_dict[current_class[0]][correct_wrong + '_image_id'] = \
                                split_[-1].split(':')[2] if '_to:' in split_[-1] else split_[-1].split(':')[0].split('_')[1]
                                images_info_dict[current_class[0]][correct_wrong + '_image_class'] = \
                                    self.config.data.class_labels[int(images_info_dict[current_class[0]][correct_wrong + '_image_id'])]
                                images_info_dict[current_class[0]][correct_wrong + '_image_class'] = \
                                    images_info_dict[current_class[0]][correct_wrong + '_image_class'][:12]

                            comma_split = split_[-1].split(',end')

                            if use_offsets: #offset != self.config.data.image_size:
                                img_pil = transforms.ToPILImage()(imgs_tensor[current_class[0] - 1, :, -offset:, -offset:])
                            else:
                                img_pil = transforms.ToPILImage()(imgs_tensor[current_class[0] - 1, :, -offset:, -offset:])

                            if use_offsets: #offset != self.config.data.image_size:
                                zoom_in = img_pil.crop((100, 135, 150, 185)).resize((110, 110))
                                zoom_in_with_border = ImageOps.expand(zoom_in, border=1, fill='red')
                                img_pil.paste(zoom_in_with_border, (112, 0))
                                img_pil.paste(
                                    ImageOps.expand(
                                        img_pil.crop((100, 135, 150, 185)), border=1, fill='red')
                                        ,(99, 134)
                                )

                            img_pil.save(os.path.join(images_folder_full,
                                                     name.replace('$', '!') + '_' + filename + f'{correct_wrong}.png'))
                            #shutil.copy(filepath,
                            #            os.path.join(images_folder_full,
                            #                         name.replace('$', '!') + '_' + filename + f'{correct_wrong}.png'))
                            #class_id = self.config.data.class_labels.index(
                            #    images_info_dict[current_class[0]][correct_wrong + '_image_class'])
                            print('comma split is', comma_split)
                            print('full split', split_)
                            image_id_curr = int(images_info_dict[current_class[0]][correct_wrong + '_image_id'])
                            print('image id is', image_id_curr)

                            dict_temp = { #float(comma_split[0].split('=')[1]),
                                         'end': model(imgs_tensor[current_class[0] - 1][:, :, -self.config.data.image_size:].unsqueeze(0).cuda()).softmax(1)[0][image_id_curr].item(), #float(comma_split[1].split('_')[0].split('=')[1]),

                                         # we assume only robust and non-robust models in dict for now
                                         #'p_model_nonrobust': float(
                                         #    probs_dict_temp[nonrobust_type][0][class_id].item()),
                                         #'p_model_robust': float(probs_dict_temp[robust_type][0][class_id].item()),
                                         #'p_RATIO': float(probs_dict_temp[RATIO_type][0][class_id].item()),
                                         #'NVAE_MSE': get_NVAE_MSE(img, NVAE_model),
                                         #'NVAE_MSE_c': get_NVAE_MSE(img, get_NVAE_class_model(class_models, class_id)),
                                         'filename': filename + f'{correct_wrong}.png',
                                         'filename_diff_abs_scaled': filename + f'{correct_wrong}_diff_abs_scaled.png',
                                         #'image_end': imgs_tensor[current_class[0] - 1][:, :, -self.config.data.image_size:]
                                         }
                            if table_type == 'afw':
                                dict_temp['start'] = model(imgs_tensor[current_class[0] - 1][:, :, :self.config.data.image_size].unsqueeze(0).cuda()).softmax(1)[0].max().item()
                            else:
                                dict_temp['start'] = model(imgs_tensor[current_class[0] - 1][:, :, :self.config.data.image_size].unsqueeze(0).cuda()).softmax(1)[0][image_id_curr].item()

                            if use_offsets: # offset != self.config.data.image_size:
                                dict_temp['img_diff'] = (name.replace('$', '!') + '_' + filename + f'{correct_wrong}_diff_abs_scaled.png', (
                                                imgs_tensor[current_class[0] - 1][:, -offset+10:self.config.data.image_size-offset_2+10, self.config.data.image_size-offset+offset_2:self.config.data.image_size]
                                                - imgs_tensor[current_class[0] - 1][:, -offset+10:-offset_2+10, -offset+offset_2:]#self.config.data.image_size:]
                                         ).sum(dim=0))
                            else:
                                dict_temp['img_diff'] = (
                                name.replace('$', '!') + '_' + filename + f'{correct_wrong}_diff_abs_scaled.png', (
                                        imgs_tensor[current_class[0] - 1][:, -offset:,
                                        self.config.data.image_size - offset:self.config.data.image_size]
                                        - imgs_tensor[current_class[0] - 1][:, -offset:, -offset:]
                                    # self.config.data.image_size:]
                                ).sum(dim=0))

                            if dict_temp['img_diff'][1].min() < min_diff_pixels:
                                min_diff_pixels = dict_temp['img_diff'][1].min()
                            if dict_temp['img_diff'][1].max() > max_diff_pixels:
                                max_diff_pixels = dict_temp['img_diff'][1].max()

                            pixels.append(dict_temp['img_diff'][1])

                            if table_type in ['inverse', 'diversity', 'afw', 'apgd', 'pgd&prior', 'diversity_ODI']:
                                if table_type == 'inverse':
                                    assert float(dict_temp['end']) >= float(prefix.split('_')[3]), f'end threshold {dict_temp["end"]} is not equal to the target {prefix.split("_")[3]}!'

                                dict_temp['norm_l1'] = float(comma_split[1].split(', l_2')[0].split('l_1: ')[1])
                                dict_temp['norm_l2'] = float(comma_split[1].split(', l_inf')[0].split('l_2: ')[1])
                            #elif table_type in ['apgd', 'pgd&prior']:
                            #    dict_temp['norm'] = float(comma_split[1].split(', l_2')[0].split('l_1: ')[1]) if 'l_1' in name else float(comma_split[1].split(', l_inf')[0].split('l_2: ')[1]) if 'l_2' in name else 'NaN'
                            #elif table_type in ['afw']:
                            #    dict_temp['norm'] = float(comma_split[1].split(', l_2')[0].split('l_1: ')[1])

                            images_info_dict[current_class[0]][prefix][correct_wrong] = dict_temp
                            keys.append((current_class[0], prefix, correct_wrong))



        for current_class_, prefix_, correct_wrong_ in keys:
            dict_temp = images_info_dict[current_class_][prefix_][correct_wrong_]
            filename_, diff = dict_temp['img_diff']
            print('filename is', filename_)
            #image_end = dict_temp['image_end']
            # Difference plots, abs value, 0-1 scaled
            #saturate_from_quantile = torch.cat(pixels, 0).flatten().quantile(0.99)
            #print('quantile is', saturate_from_quantile)
            #print('diff before', diff)
            #diff[diff >= saturate_from_quantile] = max_diff_pixels
            #print('diff after', diff)
            #assert min_diff_pixels < 0
            min_diff_pixels = -max(abs(min_diff_pixels), max_diff_pixels)
            max_diff_pixels = -min_diff_pixels
            diff_scaled = ((diff - min_diff_pixels) / (max_diff_pixels - min_diff_pixels)).clip(0, 1)
            #bwr = make_colormap(
            #    [c('blue'), c('white'), -min_diff_pixels / (max_diff_pixels - min_diff_pixels), c('white'), c('red')])
            #diff_scaled = (diff - diff.min()) / (diff.max() - diff.min())
            #img_pil = transforms.ToPILImage()(1 - diff_scaled)
            #img_pil = (1 - diff_scaled).numpy()
            cm = plt.get_cmap('seismic')
            #plt.colorbar()
            #plt.savefig('random.png')
            # Apply the colormap like a function to any array:
            colored_image = cm(diff_scaled.numpy())
            assert colored_image.max() <= 1 and colored_image.min() >= 0

            Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save(os.path.join(images_folder_full,
                                  filename_))

        print('dict')
        pretty(images_info_dict)

        centering = 'c' * (1 + len(subdir_prefixes_names_to_use))

        """
        if table_type in ['benchmark/ablation']:
            columns_names = '&'.join(r'\multicolumn{1}{C{.12\textwidth}}{' + col_name[1] + ' \par (FID:' + str(
                col_name[2]) + ') \par (FID^{T} \kern-0.5em :' + str(col_name[3]) + ')}' for col_name in subdir_prefixes_names_to_use)
        """

        if table_type in ['apgd', 'benchmark', 'ablation', 'prior']:
            columns_names = '&'.join(r'\multicolumn{1}{C{.12\textwidth}}{' + col_name[1] + '}' for col_name in subdir_prefixes_names_to_use)
        elif table_type == 'afw':
            columns_names = '&'.join(
                r'\multicolumn{1}{C{.12\textwidth}}{' + col_name[1] + '}' for col_name in subdir_prefixes_names_to_use)
        elif table_type == 'inverse':
            columns_names = '&'.join(
                r'\multicolumn{1}{C{.12\textwidth}}{' + col_name[1] + '}' for col_name in subdir_prefixes_names_to_use)
        elif table_type in ['diversity', 'diversity_ODI']:
            columns_names = '&'.join(
                r'\multicolumn{1}{C{.12\textwidth}}{' + col_name[1] + '}' for col_name in subdir_prefixes_names_to_use)
        elif table_type == 'pgd&prior':
            columns_names = '&'.join(
                r'\multicolumn{1}{C{.12\textwidth}}{' + col_name[1] + '}' for col_name in subdir_prefixes_names_to_use)

        generated_subfigures = ''
        for i in self.config.evaluation.ids:
            print(i, 'info dict', images_info_dict[i])
            # add 3.3 for each extra data row
            generated_subfigures += r'''\begin{subfigure}{0.12\textwidth}\centering
     %\vspace*{3.6mm}
     \caption*{\textbf{t}:''' + f"{images_info_dict[i]['correct_image_class']}," + "\\\\ \\textbf{s}:" + f"{images_info_dict[i]['wrong_image_class']}" + r'''}
     \includegraphics[width=1\textwidth]{''' + images_folder + '/' + 'init_' + images_info_dict[i]['init_image_path'] + r'''}
     \end{subfigure}'''
            for kwargs in subdir_prefixes_names_to_use:
                prefix = kwargs[0][0]
                name = kwargs[1]
                print('kwargs is', kwargs[0])
                ## prefix, name, FID_CIFAR10, FID_TinyImages
                ##NVAE_c = images_info_dict[i][prefix]['correct']['NVAE_MSE_c']
                ##NVAE_c = NVAE_c if type(NVAE_c) == str else f"{NVAE_c:.2f}"
                if table_type in ['inverse', 'diversity', 'afw', 'apgd', 'pgd&prior', 'diversity_ODI']:
                    print('norm is', images_info_dict[i][prefix]['correct']['norm_l1'], images_info_dict[i][prefix]['correct']['norm_l2'], prefix)
                elif table_type in ['benchmark', 'ablation', 'prior']:
                    pass
                else:
                    print('norm is', images_info_dict[i][prefix]['correct']['norm'], prefix)
                first_caption_row_dict = {
                                             'apgd': lambda: f"$l_1$/$l_2$:${images_info_dict[i][prefix]['correct']['norm_l1']:.1f}$/${images_info_dict[i][prefix]['correct']['norm_l2']:.1f}$, ",
                                             'afw': lambda: f"$l_1$/$l_2$:${images_info_dict[i][prefix]['correct']['norm_l1']:.1f}$/${images_info_dict[i][prefix]['correct']['norm_l2']:.1f}$, ",
                                             'inverse': lambda: f"$l_1$:${images_info_dict[i][prefix]['correct']['norm_l1']:.1f}$, ",
                                             'diversity': lambda: f"$l_2$:${images_info_dict[i][prefix]['correct']['norm_l2']:.1f}$, \\\\",
                                             'diversity_ODI': lambda: f"$l_2$:${images_info_dict[i][prefix]['correct']['norm_l2']:.1f}$, \\\\",
                                             'pgd&prior': lambda: f"$l_1$/$l_2$:${images_info_dict[i][prefix]['correct']['norm_l1']:.1f}$/${images_info_dict[i][prefix]['correct']['norm_l2']:.1f}$, ",
                                             'benchmark': lambda: f"$p_i$:${images_info_dict[i][prefix]['correct']['start']:.2f}$, \\\\",
                                             'prior': lambda: f"$p_i$:${images_info_dict[i][prefix]['correct']['start']:.2f}$, \\\\",
                                             'ablation': lambda: f"$p_i$:${images_info_dict[i][prefix]['correct']['start']:.2f}$, \\\\",
                }

                second_caption_row_dict = {
                    'apgd': lambda: f"$p_i$:${images_info_dict[i][prefix]['correct']['start']:.2f}$, $p_e$:${images_info_dict[i][prefix]['correct']['end']:.2f}$",
                    'afw': lambda: f"$p_e$:${images_info_dict[i][prefix]['correct']['end']:.2f}$",
                    'inverse': lambda: f"$l_2$:${images_info_dict[i][prefix]['correct']['norm_l2']:.1f}$",
                    'diversity': lambda: f"$p_i$:${images_info_dict[i][prefix]['correct']['start']:.2f}$, $p_e$:${images_info_dict[i][prefix]['correct']['end']:.2f}$",
                    'diversity_ODI': lambda: f"$p_i$:${images_info_dict[i][prefix]['correct']['start']:.2f}$, $p_e$:${images_info_dict[i][prefix]['correct']['end']:.2f}$",
                    'pgd&prior': lambda: f"$p_i$:${images_info_dict[i][prefix]['correct']['start']:.2f}$, $p_e$:${images_info_dict[i][prefix]['correct']['end']:.2f}$",
                    'benchmark': lambda: f"$p_e$:${images_info_dict[i][prefix]['correct']['end']:.2f}$",
                    'prior': lambda: f"$p_e$:${images_info_dict[i][prefix]['correct']['end']:.2f}$",
                    'ablation': lambda: f"$p_e$:${images_info_dict[i][prefix]['correct']['end']:.2f}$",

                }

                generated_subfigures += r'''&\begin{subfigure}{0.12\textwidth}\centering
     %\vspace*{3.8mm}
     \caption*{''' + first_caption_row_dict[table_type]() + \
                     second_caption_row_dict[table_type]() + r'''}
                     
     \includegraphics[width=1\textwidth]{''' + images_folder + '/' + name.replace('$', '!') + '_' + \
                                        images_info_dict[i][prefix]['correct']['filename'] + r'''}
     %\label{fig:taba}
     \end{subfigure}'''
            generated_subfigures += r'\\'

            # diff plots

            if table_type in diff_plots_tabletypes:

                for kwargs in subdir_prefixes_names_to_use:
                    prefix = kwargs[0][0]
                    name = kwargs[1]
                    generated_subfigures += r'''&\begin{subfigure}{0.12\textwidth}\centering
         \includegraphics[width=1\textwidth]{''' + images_folder + '/' + name.replace('$', '!') + '_' + \
                                            images_info_dict[i][prefix]['correct']['filename_diff_abs_scaled'] + r'''}
         %\label{fig:taba}
         \end{subfigure}'''

                generated_subfigures += r'\\'

            try:
                for kwargs in subdir_prefixes_names_to_use:
                    prefix = kwargs[0][0]
                    name = kwargs[1]

                    print(i, prefix, name)
                    ##NVAE_c = images_info_dict[i][prefix]['wrong']['NVAE_MSE_c']
                    ##NVAE_c = NVAE_c if type(NVAE_c) == str else f"{NVAE_c:.2f}"
                    first_caption_row_dict = {
                        'apgd': lambda: f"$l_1$/$l_2$:${images_info_dict[i][prefix]['wrong']['norm_l1']:.1f}$/${images_info_dict[i][prefix]['wrong']['norm_l2']:.1f}$, ",
                        'afw': lambda: f"$l_1$/$l_2$:${images_info_dict[i][prefix]['wrong']['norm_l1']:.1f}$/${images_info_dict[i][prefix]['wrong']['norm_l2']:.1f}$, ",
                        'inverse': lambda: f"$l_1$:${images_info_dict[i][prefix]['wrong']['norm_l1']:.1f}$, ",
                        'diversity': lambda: f"$l_2$:${images_info_dict[i][prefix]['wrong']['norm_l2']:.1f}$, \\\\",
                        'diversity_ODI': lambda: f"$l_2$:${images_info_dict[i][prefix]['wrong']['norm_l2']:.1f}$, \\\\",
                        'pgd&prior': lambda: f"$l_1$/$l_2$:${images_info_dict[i][prefix]['wrong']['norm_l1']:.1f}$/${images_info_dict[i][prefix]['wrong']['norm_l2']:.1f}$, ",
                        'benchmark': lambda: f"$p_i$:${images_info_dict[i][prefix]['wrong']['start']:.2f}$, \\\\",
                        'prior': lambda: f"$p_i$:${images_info_dict[i][prefix]['wrong']['start']:.2f}$, \\\\",
                        'ablation': lambda: f"$p_i$:${images_info_dict[i][prefix]['wrong']['start']:.2f}$, \\\\"
                    }

                    second_caption_row_dict = {
                        'apgd': lambda: f"$p_i$:${images_info_dict[i][prefix]['wrong']['start']:.2f}$, $p_e$:${images_info_dict[i][prefix]['wrong']['end']:.2f}$",
                        'afw': lambda: f"$p_e$:${images_info_dict[i][prefix]['wrong']['end']:.2f}$",
                        'inverse': lambda: f"$l_2$:${images_info_dict[i][prefix]['wrong']['norm_l2']:.1f}$, ",
                        'diversity': lambda: f"$p_i$:${images_info_dict[i][prefix]['wrong']['start']:.2f}$, $p_e$:${images_info_dict[i][prefix]['wrong']['end']:.2f}$",
                        'diversity_ODI': lambda: f"$p_i$:${images_info_dict[i][prefix]['wrong']['start']:.2f}$, $p_e$:${images_info_dict[i][prefix]['wrong']['end']:.2f}$",
                        'pgd&prior': lambda: f"$p_i$:${images_info_dict[i][prefix]['wrong']['start']:.2f}$, $p_e$:${images_info_dict[i][prefix]['wrong']['end']:.2f}$",
                        'benchmark': lambda: f"$p_e$:${images_info_dict[i][prefix]['wrong']['end']:.2f}$",
                        'prior': lambda: f"$p_e$:${images_info_dict[i][prefix]['wrong']['end']:.2f}$",
                        'ablation': lambda: f"$p_e$:${images_info_dict[i][prefix]['wrong']['end']:.2f}$"
                    }

                    generated_subfigures += r'''&\begin{subfigure}{0.12\textwidth}\centering
                    %\vspace*{3.8mm}
                    \caption*{''' + first_caption_row_dict[table_type]() + \
                                    second_caption_row_dict[table_type]() + r'''}
                    \includegraphics[width=1\textwidth]{''' + images_folder + '/' + name.replace('$', '!') + '_' + \
                                            images_info_dict[i][prefix]['wrong']['filename'] + r'''}
                    %\label{fig:taba}
                    \end{subfigure}'''

                    # diff plots

                if table_type in diff_plots_tabletypes:
                    generated_subfigures += r'\\'
                    for kwargs in subdir_prefixes_names_to_use:
                        prefix = kwargs[0][0]
                        name = kwargs[1]
                        generated_subfigures += r'''&\begin{subfigure}{0.12\textwidth}\centering
                         \includegraphics[width=1\textwidth]{''' + images_folder + '/' + name.replace('$', '!') + '_' + \
                                                images_info_dict[i][prefix]['wrong']['filename_diff_abs_scaled'] + r'''}
                         %\label{fig:taba}
                         \end{subfigure}'''
            except:
                print('no second class')

            generated_subfigures += r'\\ '
        # print('gener', generated_subfigures)
        content = r'''\begin{table*}[hbt!]
     \captionsetup{font=scriptsize} 
     \begin{adjustbox}{width=1\columnwidth,center}
     \centering
     \begin{tabular}{''' + f'''{centering}''' + \
                  r'''}
                   \hline
                   \multicolumn{1}{c}{Orig.} & ''' \
                  + f'''{columns_names}''' + \
                  r'''\\
                   \hline
                   ''' + generated_subfigures + r'''
     \hline
     \end{tabular}
     \end{adjustbox}
     \caption{\label{tab:Inverse_poblem}Counterfactuals of different models using PGD for a fixed radius of the $l^2$ ball $\epsilon=2.5$.}
     \end{table*}
            '''

        with open(os.path.join(images_folder_full, 'cover.tex'), 'w') as f:
            f.write(content)
