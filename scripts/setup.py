#!/usr/bin/env python3
import argparse
from datetime import datetime
from glob import glob
import logging  # noqa
import os


import shlex
import subprocess
from typing import Union, Optional
import zipfile
import tarfile
from awesome.run.config import Config
from awesome.util.done_file_marker import DoneFileMarker
from awesome.util.logging import basic_config, logger
import urllib.request

from awesome.util.path_tools import format_os_independent, relpath
from awesome.util.setup_config import SetupConfig
from tqdm.auto import tqdm
import shutil


def current_filename() -> str:
    return os.path.basename(__file__).split('.')[0]


def config():
    from awesome.mixin.argparser_mixin import set_warning_on_unsupported_type
    set_warning_on_unsupported_type(False)
    basic_config()


def win_exec(cmd: str):
    cmd_arr = ["powershell", "-C", cmd]
    logger.info(f"Executing command: \n{' '.join(cmd_arr)}")
    ret = exec_cmd(cmd)
    logger.info(f"Command executed with return code {ret[0]}")


def exec_cmd(cmd: Union[str, list]):
    arr = shlex.split(cmd) if isinstance(cmd, str) else cmd
    pop = subprocess.Popen(arr, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    pop.wait()
    return [pop.returncode, pop.communicate()[0]]

def linux_exec(cmd: str):
    logger.info(f"Executing command: \n{cmd}")
    ret = exec_cmd(cmd)
    # Log the output,
    if ret[1]:
        logger.info(f"Output: {ret[1].decode('utf-8')}")
    logger.info(f"Command executed with return code {ret[0]}")


def move_file(src: str, dest: str):
    if os.path.exists(dest) and not os.path.exists(src):
        logger.warning(f"Destination {dest} already exists, skipping.")
        return
    elif os.path.exists(dest) and os.path.exists(src):
        # Delete the destination if it exists
        logger.warning(f"Destination {dest} already exists, deleting.")
        shutil.rmtree(dest)
    os.rename(src, dest)
    logger.info(f"Moved {src} to {dest}")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, path: str, filename: str = None):
    if filename is None:
        desc = "Downloading: " + url.split('/')[-1]
    else:
        desc = "Downloading: " + filename
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=path, reporthook=t.update_to)
        t.set_description("Download completed.", refresh=True)
    logger.info(f"Download completed, saved at: {format_os_independent(path)}")


def download_git_repo(url: str, path: str):
    if os.path.exists(path):
        logger.warning(f"Path {path} already exists, skipping.")
        return
    cmd = f"git clone {url} {path}"
    if os.name == 'nt':
        win_exec(cmd)
    else:
        linux_exec(cmd)
    logger.info(f"Cloned repository from {url} to {path}")

def get_done_archive_marker(archive_path: str, target_path: str) -> str:
    file, ext = os.path.splitext(archive_path)
    filename = os.path.basename(file)
    return DoneFileMarker.get_marker_file(filename, target_path)

def extract_archive(archive_path: str, target_path: str, prefix: str = ""):
    file, ext = os.path.splitext(archive_path)
    filename = os.path.basename(file)

    with DoneFileMarker(target_path, filename) as marker:
        if marker.notify_if_exists("Skipping extraction."):
            return
        if ext == ".zip":
            with zipfile.ZipFile(archive_path) as f:
                it = tqdm(f.infolist(), desc=prefix +
                          f'Extracting {filename}:', delay=1)
                for member in it:
                    try:
                        it.set_description(
                            f"{prefix}Extracting {filename}: {member.filename}", refresh=False)
                        f.extract(member, target_path)
                    except zipfile.error as e:
                        logger.error(f"Error extracting {member.name}: {e}")
        elif ext == ".tar":
            with tarfile.TarFile(archive_path) as f:
                it = tqdm(f.getmembers(), desc=prefix +
                          f'Extracting {filename}:', delay=1)
                for member in it:
                    try:
                        it.set_description(
                            f"{prefix}Extracting {filename}: {member.name}", refresh=False)
                        f.extract(member, target_path)
                    except tarfile.TarError as e:
                        logger.error(f"Error extracting {member.name}: {e}")
        else:
            raise ValueError(f"Unknown archive type: {ext}")

    logger.info(
        f"Extraction of {filename} to {format_os_independent(target_path)} completed.")


def get_config() -> SetupConfig:
    parser = argparse.ArgumentParser(
        description='Code setup for the awesome project.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Adding a config
    parser.add_argument(
        "--config-path", type=str, default=None, required=False, help="Path to load a SetupConfig from. Can be json or yaml.")

    parser = SetupConfig.get_parser(parser)
    args = parser.parse_args()

    config: SetupConfig = None
    if args.config_path:
        args.config_path = args.config_path.strip("\"").strip("\'")
        config = SetupConfig.load_from_file(args.config_path)
        config.apply_parsed_args(args)
    else:
        config = SetupConfig.from_parsed_args(args)

    return config


def init_convexity_data(config: SetupConfig):
    if not config.download_convexity_data:
        return
    prefix = "Convexity Dataset: "
    data_url = config.get_urls().get("convexity_dataset")
    feat_url = config.get_urls().get("convexity_dataset_feat")
    # Download the data
    data_zip_path = os.path.join(config.dataset_path, "ConvexityDB.zip")
    feat_zip_path = os.path.join(config.dataset_path, "Feat.zip")
    convexity_dataset_path = os.path.join(
        config.dataset_path, "convexity_dataset")
    feat_path = os.path.join(convexity_dataset_path, "Feat")
    prog = tqdm(total=5 if not config.delete_zip_files else 6,
                desc=prefix + "Downloading data...")

    try:
        if not os.path.exists(config.dataset_path):
            os.makedirs(config.dataset_path)

        if not os.path.exists(data_zip_path) and not os.path.exists(convexity_dataset_path):
            download_file(data_url, data_zip_path)

        prog.update(1)

        prog.set_description(prefix + "Extracting data...")

        # Extract the data with zipfile
        extract_archive(data_zip_path, convexity_dataset_path, prefix)

        prog.update(1)

        prog.set_description(prefix + "Renaming folders...")
        # Rename the ground truth and user scribbles folder
        old_gt_name = os.path.join(convexity_dataset_path, "groun truth")
        new_gt_name = os.path.join(convexity_dataset_path, "ground_truth")
        if os.path.exists(old_gt_name):
            move_file(old_gt_name, new_gt_name)

        old_scrib_name = os.path.join(convexity_dataset_path, "user scribbles")
        new_scrib_name = os.path.join(convexity_dataset_path, "user_scribbles")
        if os.path.exists(old_scrib_name):
            move_file(old_scrib_name, new_scrib_name)
        prog.update(1)

        prog.set_description(
            prefix + "Download pre-calculate soft-semantic-segmentation features...")
        if not os.path.exists(feat_zip_path) and not os.path.exists(get_done_archive_marker(feat_zip_path, convexity_dataset_path)):
            download_file(feat_url, feat_zip_path, "Feat.zip")
        prog.update(1)

        prog.set_description(prefix + "Extracting features...")
        # Extract the features with zipfile
        extract_archive(feat_zip_path, convexity_dataset_path, prefix)
        prog.update(1)

        # Remove the zip file
        if config.delete_zip_files:
            prog.set_description(prefix + "Removing zip file...")
            if os.path.exists(data_zip_path):
                os.remove(data_zip_path)
            if os.path.exists(feat_zip_path):
                os.remove(feat_zip_path)
            prog.update(1)

        prog.set_description(prefix + "Done.", refresh=True)

    finally:
        prog.close()


def init_fbms_data(config: SetupConfig):
    if not config.download_fbms_data:
        return
    prefix = "FBMS-59: "
    data_url = config.get_urls().get("fbms_train_dataset")
    # Download the data
    data_zip_path = os.path.join(config.dataset_path, "FBMS_Trainingset.zip")
    fbms_dataset_path = os.path.join(config.dataset_path, "FBMS-59")
    prog = tqdm(total=4 if not config.delete_zip_files else 5,
                desc=prefix + "Downloading data...")

    mapping_url = config.get_urls().get("fbms_id_mappings")
    mapping_path = os.path.join(
        config.dataset_path, "fbms_segmentation_object_mapping.json")

    try:
        if not os.path.exists(config.dataset_path):
            os.makedirs(config.dataset_path)

        if not os.path.exists(data_zip_path) and not os.path.exists(fbms_dataset_path):
            download_file(data_url, data_zip_path)
        prog.update(1)

        prog.set_description(prefix + "Extracting fbms data...")

        # Extract the data with zipfile
        extract_archive(data_zip_path, fbms_dataset_path, prefix)

        prog.update(1)

        prog.set_description(prefix + "Renaming folder...")
        # Rename TrainingSet
        if os.path.exists(os.path.join(fbms_dataset_path, "TrainingSet")):
            move_file(os.path.join(fbms_dataset_path, "TrainingSet"),
                      os.path.join(fbms_dataset_path, "train"))
        prog.update(1)

        prog.set_description(prefix + "Downloading object mapping...")
        # Download the object mapping
        if not os.path.exists(mapping_path):
            download_file(mapping_url, mapping_path,
                          filename="fbms_segmentation_object_mapping.json")
        prog.update(1)

        # Remove the zip file
        if config.delete_zip_files:
            prog.set_description(prefix + "Removing zip file...")
            if os.path.exists(data_zip_path):
                os.remove(data_zip_path)
            prog.update(1)

        prog.set_description(prefix + "Done.", refresh=True)
    finally:
        prog.close()


def unpack_model_checkpoints(config: SetupConfig, source_dir: str, target_dir: str, prefix: str = ""):
    import torch

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = [file for file in os.listdir(source_dir) if ".pth" in file]
    mapping = {file: file.split(".")[0] + "_unet.pth" for file in files}

    missing = {x: y for x, y in mapping.items(
    ) if not os.path.exists(os.path.join(target_dir, y))}

    it = tqdm(missing.items(), desc=prefix +
              "Unpacking checkpoints...", delay=1)

    for file, name in it:
        name = file.split(".")[0] + "_unet"
        file_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, name + ".pth")

        state_dict = torch.load(
            file_path, map_location=torch.device('cpu')).get('state_dict')
        if state_dict is None:
            logger.warning(f"{prefix}Could not load {file}")
            continue
        torch.save(state_dict, target_path)


def init_uncertainty_models(config: SetupConfig):
    if not config.download_uncertainty_multicut_models:
        return
    prefix = "Uncertainty Models: "
    code_url = config.get_urls().get("multicut_uncertainty_code")

    code_path = os.path.join(
        config.third_party_code_path, "uncertainty_multicut")
    code_tar_path = os.path.join(
        config.third_party_code_path, "modelsUncertFbms.tar")

    inner_name = "FBMS59-train-masks-with-confidence-flownet2-based"
    inner_path = os.path.join(config.third_party_code_path, inner_name)

    checkpoint_path = os.path.join(
        config.checkpoint_path, "labels_with_uncertainty_flownet2_based")
    checkpoint_source_path = os.path.join(
        code_path, "with-voting", "checkpoint")

    prog = tqdm(total=5 if not config.delete_zip_files else 6,
                desc=prefix + "Downloading code...")

    try:
        # Download the code
        if not os.path.exists(config.third_party_code_path):
            os.makedirs(config.third_party_code_path)

        if not os.path.exists(code_tar_path) and not os.path.exists(code_path):
            download_file(code_url, code_tar_path,
                          filename="modelsUncertFbms.tar")
        prog.update(1)

        prog.set_description(prefix + "Extracting code...")
        # Extract the code with tar
        extract_archive(code_tar_path, config.third_party_code_path, prefix)
        prog.update(1)

        # Rename the folder to the correct name
        prog.set_description(prefix + "Renaming folder...")
        if os.path.exists(inner_path):
            move_file(inner_path, code_path)

        prog.update(1)

        # Delete __pycache__ folder
        prog.set_description(prefix + "Removing cached and compiled files...")
        cache_dir = os.path.join(code_path, "__pycache__")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        # Remove all .pyc files
        files = glob(code_path + "/**/*.pyc", recursive=True)
        for file in files:
            os.remove(file)
        prog.update(1)

        # Unpack the checkpoints
        prog.set_description(prefix + "Unpacking checkpoints...")
        unpack_model_checkpoints(
            config, checkpoint_source_path, checkpoint_path, prefix)
        prog.update(1)

        # Remove the zip file
        if config.delete_zip_files:
            prog.set_description(prefix + "Removing zip file...")
            if os.path.exists(code_tar_path):
                os.remove(code_tar_path)
            prog.update(1)

        prog.set_description(prefix + "Done.", refresh=True)

    finally:
        prog.close()


def move_and_process_labels(labels_path: str, dataset_dirs: str, prefix: str = ""):
    import h5py
    import numpy as np
    from awesome.run.functions import save_mask

    algo_name = os.path.split(labels_path)[-1]

    with DoneFileMarker(dataset_dirs, algo_name) as marker:
        if marker.notify_if_exists("Skipping label processing."):
            return
        it = tqdm(os.listdir(labels_path), desc=prefix + "Processing folders")
        h5progress = None
        try:
            for folder in it:
                it.set_description(prefix + f"Processing {folder}")
                complete_track_path = os.path.join(labels_path, folder)
                h5_files = [x for x in os.listdir(
                    complete_track_path) if x.endswith(".h5")]

                target_path = os.path.join(
                    dataset_dirs, folder, "weak_labels", algo_name)
                os.makedirs(target_path, exist_ok=True)
                confidence = None
                if h5progress is None:
                    h5progress = tqdm(total=len(h5_files),
                                      desc="Processing h5 files")
                else:
                    h5progress.reset(total=len(h5_files))
                for h5_file in h5_files:
                    path = os.path.join(complete_track_path, h5_file)
                    name = os.path.splitext(os.path.basename(h5_file))[0].replace(".ppm", "")
                    with h5py.File(path, "r") as f:
                        # 0 = background, 1 = foreground, -1 = no label
                        weak_label = np.asarray(f["img"]).T
                        confidence = np.asarray(f["confidence"]).T

                    mask = np.zeros_like(weak_label, dtype=np.uint8)

                    # Reset labels indices
                    vals = np.unique(weak_label)
                    if len(vals) == 3:
                        # Single object case
                        if (0 in vals) and (1 in vals):

                            mask[weak_label == 0] = 255
                            mask[weak_label == 1] = 1
                        else:
                            mask[...] = weak_label[...] + 1

                    else:
                        mask[...] = weak_label[...] + 1

                    save_mask(mask, os.path.join(target_path, f"{name}.png"))

                    with h5py.File(os.path.join(target_path, f"{name}_confidence.h5"), "w") as f:
                        f['confidence'] = confidence

                    h5progress.update(1)
        finally:
            if h5progress is not None:
                h5progress.close()


def init_multicut_labels(config: SetupConfig):
    if not config.download_uncertainty_multicut_labels:
        return
    prefix = "Uncertainty Labels: "
    data_url = config.get_urls().get("multicut_uncertainty_labels")

    labels_tar_path = os.path.join(config.dataset_path, "labelsUncertFbms.tar")
    fbms_dataset_path = os.path.join(config.dataset_path, "FBMS-59")
    labels_path = os.path.join(
        fbms_dataset_path, "labels_with_uncertainty_flownet2_based")
    # Check if fbms data is already downloaded
    fbms_dataset_path = os.path.join(config.dataset_path, "FBMS-59")
    fbms_dataset_train_path = os.path.join(fbms_dataset_path, "train")
    if not os.path.exists(fbms_dataset_path):
        raise FileNotFoundError(
            "FBMS-59 dataset not found, please download it first.")
    if not os.path.exists(fbms_dataset_train_path):
        raise FileNotFoundError(
            "FBMS-59 train dataset not found, please download it first.")

    prog = tqdm(total=3 if not config.delete_zip_files else 4,
                desc="Uncertainty Labels: Downloading labels...")

    try:
        # Download the labels
        if not os.path.exists(labels_tar_path) and not os.path.exists(get_done_archive_marker(labels_tar_path, fbms_dataset_path)):
            download_file(data_url, labels_tar_path,
                          filename="labelsUncertFbms.tar")
        prog.update(1)

        prog.set_description(prefix + "Extracting labels...")
        # Extract the labels with tar
        extract_archive(labels_tar_path, fbms_dataset_path, prefix)
        prog.update(1)

        # Move and process the labels
        prog.set_description(prefix + "Processing labels...")
        move_and_process_labels(labels_path, fbms_dataset_train_path, prefix)
        prog.update(1)

        # Remove the zip file
        if config.delete_zip_files:
            prog.set_description(prefix + "Removing zip file...")
            if os.path.exists(labels_tar_path):
                os.remove(labels_tar_path)
            prog.update(1)

        prog.set_description(prefix + "Done.", refresh=True)
    finally:
        prog.close()

def init_soft_semantic_segmentation_models(config: SetupConfig):
    if not config.download_soft_semantic_segmentation_models:
        return
    code_url = config.get_urls().get("soft_semantic_segmentation_code")
    models_url = config.get_urls().get("soft_semantic_segmentation_models")

    prefix = "Soft Semantic Segmentation: "

    code_path = os.path.join(config.third_party_code_path,
                             "soft_semantic_segmentation")
    checkpoint_path = os.path.join(
        config.checkpoint_path, "soft_semantic_segmentation")
    prog = tqdm(total=4 if not config.delete_zip_files else 3,
                desc=prefix + "Downloading code...")

    try:
        # Download the code
        if not os.path.exists(config.third_party_code_path):
            os.makedirs(config.third_party_code_path)

        if not os.path.exists(code_path):
            download_git_repo(code_url, code_path)
        prog.update(1)

        # Upgrade code to tf2 with upgrade script
        prog.set_description(prefix + "Downloading models...")
        zip_file_path = os.path.join(config.checkpoint_path, "SSS_model.zip")
        if not os.path.exists(zip_file_path) and not os.path.exists(get_done_archive_marker(zip_file_path, checkpoint_path)):
            download_file(models_url, zip_file_path, filename="SSS_model.zip")
        prog.update(1)

        prog.set_description(prefix + "Extracting models...")
        # Extract the code with zipfile
        extract_archive(zip_file_path, checkpoint_path, prefix)
        prog.update(1)

        if config.delete_zip_files:
            prog.set_description(prefix + "Removing zip file...")
            if os.path.exists(zip_file_path):
                os.remove(zip_file_path)
            prog.update(1)

        prog.set_description(prefix + "Done.", refresh=True)

    finally:
        prog.close()


def init_pc_pretrain_states(config: SetupConfig):
    if not config.download_pc_pretrain_states:
        return
    prefix = "Path-Connected Pretrained States: "
    data_url = config.get_urls().get("pc_pretrain_states")

    data_zip_path = os.path.join(
        config.checkpoint_path, "pc_pretrain_states.zip")
    pc_states_path = os.path.join(config.checkpoint_path, "pretrain_states")

    prog = tqdm(total=2 if not config.delete_zip_files else 3,
                desc=prefix + "Downloading data...")

    try:
        if not os.path.exists(config.checkpoint_path):
            os.makedirs(config.checkpoint_path)

        if not os.path.exists(data_zip_path) and not os.path.exists(pc_states_path):
            download_file(data_url, data_zip_path)
        prog.update(1)

        prog.set_description(prefix + "Extracting data...")
        # Extract the data with zipfile
        extract_archive(data_zip_path, pc_states_path, prefix)
        prog.update(1)

        # Remove the zip file
        if config.delete_zip_files:
            prog.set_description(prefix + "Removing zip file...")
            if os.path.exists(data_zip_path):
                os.remove(data_zip_path)
            prog.update(1)

        prog.set_description(prefix + "Done.", refresh=True)

    finally:
        prog.close()


def main(config: SetupConfig):
    init_convexity_data(config)
    init_fbms_data(config)
    init_uncertainty_models(config)
    init_multicut_labels(config)
    init_soft_semantic_segmentation_models(config)
    init_pc_pretrain_states(config)


if __name__ == "__main__":
    config()
    cfg = get_config()
    try:
        main(cfg)
    except Exception as err:
        logging.exception(
            f"Raised {type(err).__name__} in {current_filename()}, exiting...")
        exit(1)
    exit(0)
