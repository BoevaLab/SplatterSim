[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# MalSim: Simulation for malignant cells with joint CNV profile 

A pipeline for simulating single cancer cells with joint copy number variation (CNV) profiles. 
This simulation is an adaptation of the [Splatter](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1305-0) algorithm that includes the effect of CNVs on the expression profile of cells. 

_Note: This repository is in the experimental stage. Changes to the API may appear._

### Description of the parameters
- `anchors`: list of size n_states that chooses a gene that, when gained, increases the chance of the cell to be in the program. If you are simulating 2 states, and anchors are `["geneA","geneB"]`, then a cell with a gain on geneA will be more likely to be in state 1
- `genome`: this is the genome information to create CNVs. In our case, the gene names themselves don't mean anything anymore in terms of function/pathways, they are just used to be able to map genomic positions. 
- `chromosomes_gain`: the list of chromosomes on which it is possible to have a gain for a clone/subclone. 
- `chromosomes_loss`: the list of chromosomes on which it is possible to have a loss for a clone/subclone. Pay attention to the fact that the anchors are gained independently, so if the chromosomes on which a loss is possible include the chromosomes where the anchors are, you might get an improbable situation of a gain in the middle of a loss.
- `dropout`: probability of a chromosome of the list not being gained/lost. This means that if the dropout is 0.7, then on average 30% of the chromosomes of the list chromosomes_gain/loss will be gained/lost. In effect, we iterate of the chromosomes_gain/loss list and draw from a Bernouilli(1-dropout) if an event will happen on the chromosome.
- `dropout_child`: same as dropout, but applies to subclones. Keep in mind for generating subclones, we start off with the ancestral CNV profile and generate additional events on top of it. The list of possible chromosomes where an event can happen is reduced to those that haven't seen an event yet. 
- `p_anchor`: probability of gaining a small region around an anchor. This is done independently of generating gains/losses on the other chromosomes. It is also done independently for every subclone. If the anchor was already gained in the ancestral clone, then the anchor remains gained no matter what.
- `min_region_length`: the minimal size of an event in terms of genes
- `max_region_length`: the maximal size of an event in terms of genes
- `seed`: the random seed used for all generation
- `n_batches`: the number of patients in the dataset to generate
- `n_programs`: the number of states to generate 
- `n_subclones_min/max`: the minimal/maximal number of subclones a patient can have. This number is picked uniformly at random between min and max.
- `n_malignant_min/max`: the minimal/maximal number of malignant cells per patient
- `n_healthy_min/max`: the minimal/maximal number of healthy cells per patient
- `subclone_alpha`: the proportions of cells attributed to each subclone in the patient is sampled from a Dirichlet with all alpha parameters equal to subclone_alpha
- `alpha_add`: when a subclone has a gain on an anchor, then the probability of a cell belonging to this subclone being of the state associated with the anchor should be higer. To model this, we start with a beginning parameterization of a Dirichlet distribution (see `start_alpha`) and if a gain is present on the anchor, we add alpha_add to the corresponding alpha. This means that different subclones with different anchor profiles will have different starting dirichlet distributions to sample the proability for a cell of the subclones to belong to each state. 
- `start_alpha`: list of the same size as the number of states simulated, `start_alpha[i]` is the alpha for the dirichlet distribution associated with state i if there are no gains on the anchor. 
- `celltypes`: list of the names of the celltypes simulated. Beware that in the current version, the healthy cells MUST be called "Macro" and "Plasma". The simulated malignant programs can be called anything. The other parameters will be associated in the same order, i.e., if celltypes is `["Macro", "Plasma", "state1", "state2"]`, then then all positions 0 in the next parameters will refer to the "Macro" cells, all positions 1 to the "Plasma", etc.
- `p_drop`: list of size n_states - 1, with the probability of dropping the rarest program in position 0, the second rarest program given the first has been dropped in position 1, etc.
- `batch_effect`: boolean, if True, then batch effect is added
- `libsize_scale/loc`: the library size is sampled from a log-normal distribution (see Splatter paper for more details). These parameters are the scale/location of the log-normal distribution to sample from.
- `p_de_list`: list of size n_celltypes, containing the probability of a gene to be differentially expressed in this celltype/state (see Splatter paper for more details).
- `p_down_list`: list of size n_celltypes, containing the probability of a gene that is differentially expressed to be downregulated in this celltype/state (see Splatter paper for more details).
- `de_location/scale_list`: list of size n_celltypes, containing the location and scale factor for each cell type for the differential expression parameters (see Splatter paper for more details).
- `pb_de_list`: probability of genes to be differentially expressed linked to the batch. 
- `bde_location/scale_list`: location/scale for the differential expression parameters linked to the batch (see Splatter paper for more details).
- `shared_cnv`: boolean, if True, the effect of a gene being gained/lost will be shared across all patients, i.e. if patient 1 and patient 2 both have a gain on geneA, then the modification on the gene expression will be the same. if False, then although the effect of a gain/loss is shared across cells of the same patient, they are different across patients.
