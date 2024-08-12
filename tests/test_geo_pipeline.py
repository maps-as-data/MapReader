from __future__ import annotations

from pathlib import Path

import pytest

from mapreader import (
    AnnotationsLoader,
    ClassifierContainer,
    PatchDataset,
    SheetDownloader,
    load_patches,
    loader,
)


@pytest.fixture
def sample_dir():
    return Path(__file__).resolve().parent / "sample_files"


def test_pipeline(tmp_path, sample_dir):
    # Download
    print(f"{sample_dir}/test_json.json")
    my_ts = SheetDownloader(
        metadata_path=f"{sample_dir}/test_json.json",
        download_url="https://mapseries-tilesets.s3.amazonaws.com/1inch_2nd_ed/{z}/{x}/{y}.png",
    )

    my_ts.extract_wfs_id_nos()
    my_ts.get_grid_bb(14)
    my_ts.download_map_sheets_by_wfs_ids(
        [131, 132], path_save=f"{tmp_path}/maps", force=True
    )

    # Load
    my_files = loader(f"{tmp_path}/maps/*png")
    my_files.add_metadata(f"{tmp_path}/maps/metadata.csv")

    my_files.patchify_all(
        patch_size=300,  # in pixels
        path_save=f"{tmp_path}/patches_300_pixel",
    )

    my_files.calc_pixel_stats()
    parent_df, patch_df = my_files.convert_images()

    # skip annotate (load premade annotations instead)
    # annotations are only for map 1

    # Classify - Train
    annotated_images = AnnotationsLoader()
    annotated_images.load(
        f"{sample_dir}/land_#rw#.csv", images_dir=f"{tmp_path}/patches_300_pixel"
    )

    annotated_images.create_datasets(frac_train=0.7, frac_val=0.2, frac_test=0.1)
    dataloaders = annotated_images.create_dataloaders(batch_size=8)

    my_classifier = ClassifierContainer(
        model="resnet18", dataloaders=dataloaders, labels_map={0: "No", 1: "rail space"}
    )

    my_classifier.add_loss_fn("cross-entropy")
    params_to_optimize = my_classifier.generate_layerwise_lrs(
        min_lr=1e-4, max_lr=1e-3, spacing="geomspace"
    )
    my_classifier.initialize_optimizer(params2optimize=params_to_optimize)
    my_classifier.initialize_scheduler()

    my_classifier.train(
        num_epochs=10,
        save_model_dir=f"{tmp_path}/models",
        tensorboard_path=f"{tmp_path}/tboard",
        tmp_file_save_freq=2,
        remove_after_load=False,
        print_info_batch_freq=5,
    )

    # Classify - Inference
    my_maps = load_patches(
        f"{tmp_path}/patches_300_pixel/*102352861*png",
        parent_paths=f"{tmp_path}/maps/map_102352861.png",
    )

    my_maps.add_metadata(f"{tmp_path}/maps/metadata.csv", ignore_mismatch=True)
    my_maps.add_metadata(patch_df, tree_level="patch", ignore_mismatch=True)
    parent_df, patch_df = my_maps.convert_images()

    patch_dataset = PatchDataset(patch_df, transform="val")

    my_classifier.load_dataset(
        patch_dataset, set_name="patches", batch_size=8, shuffle=False
    )

    my_classifier.inference(set_name="patches")
