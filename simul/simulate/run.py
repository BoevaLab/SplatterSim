from typing import Dict, Union, List, Tuple

import anndata as ad
import pandas as pd
import numpy as np

from simul.base.config import SimCellConfig
import simul.base.splatter as splatter
import simul.simulate.utils as utils

from simul.patients.dataset import Dataset
from simul.cnv.sampling import ProgramDistribution
import simul.patients.create_dataset as ds


def simulate_full_obs(
    dataset: Dataset, prob_dist: ProgramDistribution, p_drop: Union[List[float], np.ndarray, float] = 0.3
) -> Dict[str, np.ndarray]:
    if type(p_drop) is float:
        p_drop = [p_drop, p_drop]
    all_malignant_obs = ds.simulate_malignant_comp_batches(dataset=dataset, prob_dist=prob_dist)
    all_malignant_obs, dataset = ds.drop_rarest_program(all_malignant_obs, dataset, p_1=p_drop[0], p_2=p_drop[1])
    all_healthy_obs = ds.simulate_healthy_comp_batches(dataset=dataset)
    full_obs = ds.get_full_obs(all_malignant_obs=all_malignant_obs, all_healthy_obs=all_healthy_obs)

    return full_obs


def get_common_mean_pc(
    config: SimCellConfig, full_obs: Dict[str, pd.DataFrame], rng: np.random.Generator
) -> np.ndarray:

    print("Sampling original mean...")
    gene_mean = splatter.sample_mean(rng=rng, shape=config.mean_shape, scale=config.mean_scale, size=(config.n_genes,))

    print("Sampling outlier factors...")
    outlier, outlier_factor = splatter.sample_outlier(
        rng=rng, p=config.p_outlier, location=config.outlier_loc, scale=config.outlier_scale, size=(config.n_genes,)
    )

    print("Changing mean to fit outliers...")
    modif_mean = splatter.transform_mean(mean=gene_mean, outlier=outlier, outlier_factor=outlier_factor)

    print("Getting per cell means...")
    mean_pp = splatter.get_mean_pp(mean=modif_mean, full_obs=full_obs)

    return mean_pp


def get_group_cnv_transformed(
    mean_pp: np.ndarray,
    full_obs: Dict[str, np.ndarray],
    config: SimCellConfig,
    dataset: Dataset,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray]]:

    print("Transforming mean linked to groups...")
    transformed_means, de_facs_groups = utils.transform_means_by_facs(
        rng=rng, config=config, full_obs=full_obs, mean_pc_pp=mean_pp, group_or_batch="group"
    )

    if config.batch_effect:
        print("Transforming mean linked to batch effect...")
        transformed_means, de_facs_be = utils.transform_means_by_facs(
            rng=rng, config=config, full_obs=full_obs, mean_pc_pp=transformed_means, group_or_batch="batch"
        )
    else:
        de_facs_be = {patient: pd.Series([]) for patient in transformed_means}

    print("Transforming mean linked to CNV profile...")
    transformed_means, gain_expr_full, loss_expr_full = utils.transform_malignant_means(
        full_obs=full_obs, transformed_means=transformed_means, dataset=dataset, shared_cnv=config.shared_cnv
    )

    return transformed_means, de_facs_groups, de_facs_be, gain_expr_full, loss_expr_full


def adjust_libsize(
    rng: np.random.Generator, transformed_means: Dict[str, np.ndarray], config: SimCellConfig
) -> Dict[str, np.ndarray]:
    print("Sampling cell-specific library size...")
    pat_libsize = splatter.sample_library_size(
        rng=rng, transformed_means=transformed_means, location=config.libsize_loc, scale=config.libsize_scale
    )
    print("Adjusting for library size...")
    libsize_means = splatter.libsize_adjusted_means(means=transformed_means, libsize=pat_libsize)
    return libsize_means


def sample_counts_patient(mean_pc: np.ndarray, config: SimCellConfig, rng: np.random.Generator) -> np.ndarray:

    print("Getting BCV...")
    bcv = splatter.sample_BCV(rng=rng, means=mean_pc, common_disp=config.common_disp, dof=config.dof)

    print("Calculating trended mean...")
    trended_mean = splatter.sample_trended_mean(rng=rng, means=mean_pc, bcv=bcv)

    print("Sampling true counts...")
    true_counts = splatter.sample_true_counts(rng=rng, means=trended_mean)

    print("Computing gene and cell-specific dropout probability...")
    dropout_prob = splatter.get_dropout_probability(
        means=trended_mean, midpoint=config.dropout_midpoint, shape=config.dropout_shape
    )

    print("Sampling dropout...")
    dropout = splatter.sample_dropout(rng=rng, dropout_prob=dropout_prob)

    print("Transforming counts with dropout...")
    counts = splatter.get_counts(true_counts=true_counts, dropout=dropout)

    return counts


def simulate_dataset(
    config: SimCellConfig, rng: np.random.Generator, full_obs: Dict[str, pd.DataFrame], dataset: Dataset
) -> Tuple[Dict[str, np.ndarray]]:
    mean_pp = get_common_mean_pc(config=config, full_obs=full_obs, rng=rng)
    transformed_means, de_facs_group, de_facs_be, gain_expr_full, loss_expr_full = get_group_cnv_transformed(
        mean_pp=mean_pp, full_obs=full_obs, config=config, dataset=dataset, rng=rng
    )
    transformed_means = adjust_libsize(rng=rng, transformed_means=transformed_means, config=config)
    final_counts_pp = {}
    for pat in transformed_means:
        final_counts_pp[pat] = sample_counts_patient(mean_pc=transformed_means[pat], config=config, rng=rng)

    return final_counts_pp, de_facs_group, de_facs_be, gain_expr_full, loss_expr_full


def counts_to_adata(
    counts_pp: Dict[str, np.ndarray], observations: Dict[str, pd.DataFrame], var: pd.DataFrame
) -> ad.AnnData:
    adatas = []
    for pat in counts_pp:

        obs = observations[pat]
        sample_df = pd.DataFrame(np.array([pat] * obs.shape[0]), index=obs.index, columns=["sample_id"])
        obs = pd.concat([obs, sample_df], axis=1)

        adatas.append(ad.AnnData(counts_pp[pat], obs=obs, var=var, dtype=counts_pp[pat].dtype))
    return adatas
