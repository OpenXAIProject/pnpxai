import argparse


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def get_arguments(parser=None) -> argparse.Namespace:
    if parser is None:
        parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument(
        "-p", "--prompt", type=str, help="The prompt for the desired editing", required=False
    )
    #parser.add_argument(
    #    "-i", "--init_image", type=str, help="The path to the source image input", required=True
    #)
    #parser.add_argument("--mask", type=str, help="The path to the mask to edit with", default=None)

    # Diffusion
    parser.add_argument(
        "--skip_timesteps",
        type=int,
        help="How many steps to skip during the diffusion.",
        default=-1,
    )
    parser.add_argument(
        "--local_clip_guided_diffusion",
        help="Indicator for using local CLIP guided diffusion (for baseline comparison)",
        action="store_true",
        dest="local_clip_guided_diffusion",
    )

    # For more details read guided-diffusion/guided_diffusion/respace.py
    parser.add_argument(
        "--timestep_respacing",
        type=str,
        help="How to respace the intervals of the diffusion process (number between 1 and 1000).",
        default='-1',
    )
    parser.add_argument(
        "--model_output_size",
        type=int,
        help="The resolution of the outputs of the diffusion model",
        default=256,
        choices=[256, 512],
    )

    # Augmentations
    parser.add_argument("--aug_num", type=int, help="The number of augmentation", default=1)

    # Loss
    parser.add_argument(
        "--clip_guidance_lambda",
        type=float,
        help="Controls how much the image should look like the prompt",
        default=-1
    )
    parser.add_argument(
        "--range_lambda",
        type=float,
        help="Controls how far out of range RGB values are allowed to be",
        default=-1
    )
    parser.add_argument(
        "--lpips_sim_lambda",
        type=float,
        help="The LPIPS similarity to the input image",
        default=-1
    )
    parser.add_argument(
        "--l2_sim_lambda", type=float, help="The L2 similarity to the input image", default=-1
    )

    parser.add_argument(
        "--l1_sim_lambda", type=float, help="The L1 similarity to the input image", default=0,
    )
    parser.add_argument(
        "--TV_lambda", type=float, help="The TV similarity to the input image", default=0,
    )
    parser.add_argument(
        "--lp_custom", type=float, help="The custom lp norm", default=-1,
    )
    parser.add_argument(
        "--ilvr_multi", type=float, help="The ilvr multiplier", default=0,
    )
    parser.add_argument(
        "--lp_custom_value", type=float, help="The value of the custom lp norm", default=-1,
    )

    parser.add_argument(
        "--layer_reg", type=float, help="The custom layers regularization", default=0,
    )

    parser.add_argument(
        "--layer_reg_value", type=float, help="The custom layers regularization", default=0,
    )

    parser.add_argument(
        "--background_preservation_loss",
        help="Indicator for using the background preservation loss",
        action="store_true",
    )

    parser.add_argument(
        "--not_use_init_image",
        help="Indicator for starting from noise",
        action="store_true",
    )


    parser.add_argument(
        "--enforce_same_norms",
        help="Indicator for enforcing the same norms for all terms",
        action="store_true",
    )

    parser.add_argument(
        "--denoise_dist_input",
        help="Retain graph after classifier",
        action="store_true",
    )

    parser.add_argument(
        "--projecting_cone",
        help="Indicator for enforcing the cone projection",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        help="Indicator for enforcing the cone projection",
        action="store_true",
    )

    parser.add_argument(
        "--use_blended",
        help="Indicator for using the blended diffusion like setting",
        action="store_true",
    )

    # Mask
    parser.add_argument(
        "--invert_mask",
        help="Indicator for mask inversion",
        action="store_true",
        dest="invert_mask",
    )
    parser.add_argument(
        "--no_enforce_background",
        help="Indicator disabling the last background enforcement",
        action="store_false",
        dest="enforce_background",
    )

    # Misc
    parser.add_argument("--seed", type=int, help="The random seed", default=-1)
    parser.add_argument("--gpu_id", type=int, help="The GPU ID", default=0)
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="The filename to save, must be png",
        default="output.png",
    )
    parser.add_argument("--iterations_num", type=int, help="The number of iterations", default=1)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="The number number if images to sample each diffusion process",
        default=-1,
    )
    parser.add_argument(
        "--vid",
        help="Indicator for saving the video of the diffusion process",
        action="store_true",
        dest="save_video",
    )
    parser.add_argument(
        "--export_assets",
        help="Indicator for saving raw assets of the prediction",
        action="store_true",
        dest="export_assets",
    )

    parser.add_argument('--gpu', '--list', nargs='+', default=[0],
                        help='GPU indices, if more than 1 parallel modules will be called')

    parser.add_argument(
        "--classifier_lambda", type=float, help="The coefficient used for the classifier", default=-1,
    )

    parser.add_argument(
        "--mssim_lambda", type=float, help="The coefficient used for the mssim", default=0,
    )

    parser.add_argument(
        "--ssim_lambda", type=float, help="The coefficient used for the mssim", default=0,
    )

    parser.add_argument(
        "--quantile_cut", type=float, help="The quantile used for cutting images at every step", default=0,
    )
    parser.add_argument(
        "--gen_type", type=str, help='Type of generation (p_sample/ddim)', default='p_sample'
    )
    parser.add_argument(
        "--method", type=str, help='Method to use (dvces/svces)', default='-1'
    )
    parser.add_argument(
        "--config", type=str, help='Config to use (default/blended/svce)', default='default.yml'
    )

    # Spurious features

    parser.add_argument(
        "--class_id_spurious",
        type=int,
        help="Id of the spurious class to generate.",
        default=-1,
    )

    parser.add_argument(
        "--component_idx_spurious",
        type=int,
        help="Id of the spurious component to generate.",
        default=-1,
    )

    parser.add_argument(
        "--start_img_id",
        type=int,
        help="Id of the starting init image.",
        default=-1,
    )

    parser.add_argument(
        "--pca_component_lambda", type=float, help="The coefficient used for the pca_component", default=0,
    )

    defaults = dict(
        classifier_size_1=224,
        classifier_size_2=224,
        classifier_size_3=224,
        target_class=-1,
        dataset='imagenet',
        data_folder='',
        project_folder='.',
        consistent=False,
        step_lr=-1,
        nsigma=1,
        model_types=None,
        ODI_steps=-1,
        fid_num_samples=1,
        begin_ckpt=1,
        end_ckpt=1,
        adam=False,
        D_adam=False,
        D_steps=0,
        model_epoch_num=0,
        device_ids=None,
        script_type='sampling',
        num_imgs=2048,
        range_t=0,
        down_N=32,
        eps_project=30,
        classifier_type=-1,
        second_classifier_type=-1,
        third_classifier_type=-1,
        deg_cone_projection=-1,
        interpolation_int_1=3,
        interpolation_int_2=3,
        interpolation_int_3=3,
        plot_freq=5,
        world_size=1,
        world_id=0,
        variance=1.0
    )

    add_dict_to_argparser(parser, defaults)

    args = parser.parse_args("") # for running in interactive mode
    # args = parser.parse_args() # for running in terminal
    
    print(args)

    return args
