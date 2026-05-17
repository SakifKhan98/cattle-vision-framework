**Cattle Vision Framework**

**Phase 0: Dataset Analysis & Label Design Report**

*CBVD-5 & CVB Datasets \| Frozen Label Taxonomy \| March 2026*

**1. Overview**

Phase 0 establishes the foundational decisions that govern every
subsequent stage of the Cattle Vision Framework. This phase covers the
selection and analysis of training datasets, the design of a unified
behavior taxonomy across heterogeneous data sources, and the freezing of
a canonical label map that all training, evaluation, and reporting code
depends on. These decisions are made before any model training and are
treated as immutable --- changing them mid-project would invalidate all
trained models and accumulated results.

**2. Dataset Selection and Rationale**

Two publicly available cattle behavior datasets were selected to provide
complementary coverage of real-world farm environments. Using a single
dataset would limit the generalizability of trained models and prevent
cross-domain evaluation, which is a core scientific contribution of this
thesis. The two datasets differ in breed, environment, camera type,
lighting, and annotation vocabulary, making them a rigorous testbed for
domain generalization.

**2.1 CBVD-5 (Li et al., 2024)**

The Cattle Behavior Video Dataset (CBVD-5) was collected at an indoor
dairy barn in China using seven fixed Dahua IP surveillance cameras. The
dataset captures 107 Holstein-Friesian dairy cows across 96 hours of
footage, producing 687 ten-second video clips at 30 fps (206,100 frames
total). Video is encoded in H.265/HEVC at 1920×1080 resolution.

  ----------------------- ---------------------------------------------------------
       **Property**                              **Detail**

        **Setting**           Indoor dairy barn, controlled lighting including
                                                  nighttime

        **Animals**                   107 Holstein-Friesian dairy cows

        **Cameras**                  7 fixed Dahua surveillance cameras

     **Total footage**      96 hours \| 687 clips × 10 seconds × 30 fps = 206,100
                                                   frames

  **Annotated keyframes**    4,122 keyframes selected → 27,501 labeled instances

   **Annotation format**     AVA-style CSV: video_id, timestamp, x1, y1, x2, y2,
                                      action_id, target_id (no header)

    **Box coordinates**       Normalized (0--1); must be converted to pixels by
                                         multiplying by 1920 / 1080

      **Multi-label**      Yes --- same box appears multiple times with different
                                    action_ids (e.g., lying + ruminating)

    **Official split**         3-way: ava_train_v2.1.csv / ava_val_v2.1.csv /
                                              ava_test_v2.1.csv

     **Image naming**      labelframes/labelframes/{video_id}\_{timestamp:05d}.jpg
                                             e.g. 618_00002.jpg
  ----------------------- ---------------------------------------------------------

**2.1.1 CBVD-5 Class Distribution**

The five behavior classes in CBVD-5 are highly imbalanced, with Standing
dominating and Drinking severely underrepresented:

  -------- --------------------- --------------- --------------- ----------------
   **ID**        **Class**           **Est.      **% of Total**   **Difficulty**
                                   Instances**                   

     1           Standing           \~943,800          41%             Low

     2             Lying            \~570,000          25%             Low

     3           Foraging           \~369,780          16%            Medium

     4          Rumination          \~388,800          17%             High

     5        Drinking water        \~43,980           2%           Very High
  -------- --------------------- --------------- --------------- ----------------

The Drinking class at 2% of instances is a known challenge in this
dataset. Li et al. (2024) reported only 34.8% test AP for the Drinking
class --- the lowest among all five behaviors. Class weighting will be
applied during VideoMAE behavior training in Phase 6 to address this
imbalance.

**2.2 CVB (Zia et al., 2023)**

The Cattle Visual Behaviors (CVB) dataset was collected on an outdoor
research pasture in Australia using four GoPro Hero cameras mounted at
the corners of a 25m × 25m paddock. The dataset captures 8 Angus beef
cattle across 502 fifteen-second video clips at 30 fps (225,900 frames
total), annotated at the per-frame level using CVAT with subsequent
correction.

  ----------------------- ------------------------------------------------------------------------------------
       **Property**                                            **Detail**

        **Setting**                          Outdoor 25m×25m pasture, natural daylight only

        **Animals**                          8 Angus beef cattle with individual track IDs

        **Cameras**                          4 GoPro cameras at field corners (wide-angle)

     **Total footage**                 502 clips × 15 seconds × 30 fps = 225,900 annotated frames

    **Annotated boxes**                   136,598 bounding boxes with per-frame per-cow labels

   **Annotation format**    AVA-style CSV: video_id (full clip name), timestamp, x1, y1, x2, y2, action_id,
                                                               animal_id

       **Track IDs**       Column H = animal_id (ground truth cow identity) --- used for tracking evaluation

    **Official split**                  80:20 train/val --- ava_train_set.csv / ava_val_set.csv

     **Image naming**                 raw_frames/{clip_id}/img\_{frame:05d}.jpg e.g. img_00001.jpg

      **Clip naming**      {clip_num}\_arm{arm}\_gopro{cam}\_{date}\_{time}\_beh{n}\_ani{n}\_ins{n}\_cut\_{n}
  ----------------------- ------------------------------------------------------------------------------------

**2.2.1 CVB Class Distribution**

CVB has a richer 12-class vocabulary including behaviors absent from
CBVD-5. Three classes are dropped from training (see Section 4):

  -------- --------------------- --------------- --------------- -------------
   **ID**      **CVB Class**      **Approx. %**    **Status**     **Maps To**

     2            Grazing              39%       **✓ Included**    Foraging

     7         Resting-lying           15%       **✓ Included**      Lying

     6       Resting-standing          12%       **✓ Included**    Standing

     1       None (occluded /          10%         --- Dropped        ---
                 unclear)                                        

     5       Ruminating-lying          6%        **✓ Included**   Ruminating

     3            Walking              5%          --- Dropped        ---

     8           Drinking              4%        **✓ Included**    Drinking

     4      Ruminating-standing        4%        **✓ Included**   Ruminating

     9           Grooming              3%        **✓ Included**    Grooming

     10            Other               2%        **✓ Included**      Other

     12           Running             \<1%         --- Dropped        ---

     11           Hidden              \<1%         --- Dropped        ---
  -------- --------------------- --------------- --------------- -------------

**3. Dataset Comparison**

The two datasets are intentionally complementary, covering the two most
common cattle farming contexts worldwide. Their differences make them a
strong testbed for domain generalization experiments.

  --------------------- ------------------------ ------------------------
      **Dimension**            **CBVD-5**                **CVB**

      **Farm type**     Intensive dairy (indoor)    Extensive pasture
                                                        (outdoor)

      **Herd size**      107 cows (large herd)     8 cows (small herd)

   **Individual IDs**        Not available       Available (ground truth
                                                         tracks)

   **Occlusion level**   High (crowded stalls)      Lower (open field)

    **Label density**   Keyframes only (sparse)    Every frame (dense)

       **Behavior              5 classes          12 classes (3 dropped)
      vocabulary**                               

      **Multi-label        Yes (e.g. lying +     No (single label per cow
      annotations**           ruminating)               per frame)

    **Use in tracking   No (no ground truth IDs) Yes (IDF1, IDSW metrics)
         eval**                                  
  --------------------- ------------------------ ------------------------

**4. Unified Behavior Taxonomy Design**

A central challenge of this project is that the two datasets use
different behavior vocabularies. CBVD-5 has 5 classes designed around a
dairy barn context; CVB has 12 classes from a pasture context. To enable
joint training and cross-dataset evaluation, a unified 7-class taxonomy
was designed that maximizes semantic overlap while being biologically
meaningful.

**4.1 Dropped CVB Classes and Rationale**

  --------------- -------------------------------------------------------
     **Dropped                         **Rationale**
      Class**     

    **Walking**     No CBVD-5 equivalent. Dairy barn cows have severely
                     restricted movement; walking is not a meaningful
                   behavior in that context. Including it would create a
                    CVB-only class with no cross-dataset comparability.

    **Running**    Fewer than 1% of CVB instances. Insufficient data to
                  train a reliable classifier. Rare enough to be subsumed
                      into Other without meaningful information loss.

    **Hidden**    Represents occlusion, not an actual behavior. Including
                      it would train the model to classify visibility
                  conditions rather than animal actions. Occluded cattle
                    are handled at the tracking stage, not the behavior
                                          stage.

      **None      Represents ambiguous or unclear annotations in CVB. Not
  (label_id: 1)** a defined behavior category. All rows with action_id=1
                            are dropped during data conversion.
  --------------- -------------------------------------------------------

**4.2 Frozen 7-Class Taxonomy**

The final canonical label set consists of 7 classes organized into three
tiers: 5 core classes that exist in both datasets and support
cross-dataset evaluation, 1 auxiliary class (Grooming) present only in
CVB, and 1 residual class (Other) for CVB behaviors not mapped
elsewhere.

  -------- --------------- ----------- ------------------- -----------------------
   **ID**    **Canonical    **Tier**    **CBVD-5 Source**      **CVB Source**
               Class**                                     

     0        Standing        Core          Standing          Resting-standing

     1          Lying         Core            Lying             Resting-lying

     2        Foraging        Core          Foraging               Grazing

     3        Drinking        Core       Drinking water           Drinking

     4       Ruminating       Core         Rumination       Ruminating-standing +
                                                              Ruminating-lying

     5        Grooming      Auxiliary     --- (absent)            Grooming

     6          Other       Residual      --- (absent)              Other
  -------- --------------- ----------- ------------------- -----------------------

The Ruminating class merges CVB's two rumination variants
(Ruminating-standing and Ruminating-lying) into a single canonical
class. This is biologically justified: rumination is the behavior of
interest, and the postural component (standing vs. lying) is already
captured by the Standing and Lying classes. A cow simultaneously lying
and ruminating would be annotated as Lying in CBVD-5 (via multi-label)
and Ruminating-lying in CVB. The merge produces consistent semantics
across both datasets.

**5. Cross-Dataset Evaluation Scope**

Cross-dataset evaluation --- a core scientific contribution of this
thesis --- is restricted to the **5 core classes (IDs 0--4)**. Grooming
(ID 5) and Other (ID 6) are excluded from cross-dataset metrics because
they exist only in CVB. Including them would create asymmetric
comparisons where CBVD-5 models have no training data for these classes.

  --------------------- --------------- --------------- ------------------
        **Class**        **In CBVD-5**    **In CVB**     **Cross-dataset
                                                              eval**

        Standing           **✓ Yes**       **✓ Yes**        **✓ Yes**

          Lying            **✓ Yes**       **✓ Yes**        **✓ Yes**

        Foraging           **✓ Yes**       **✓ Yes**        **✓ Yes**

        Drinking           **✓ Yes**       **✓ Yes**        **✓ Yes**

       Ruminating          **✓ Yes**       **✓ Yes**        **✓ Yes**

        Grooming            --- No         **✓ Yes**          --- No

          Other             --- No         **✓ Yes**          --- No
  --------------------- --------------- --------------- ------------------

**6. Implementation**

The label map is implemented in two frozen files at the project root.
These files are never modified after Phase 0.

**6.1 data/labels.yaml**

A human-readable YAML file containing the full class list with IDs and
categories, per-dataset mapping rules, dropped label lists, and the
cross-dataset evaluation scope. This is the primary reference document
for anyone working with the codebase.

**6.2 data/label_map.json**

A machine-readable JSON file used at runtime by all conversion scripts
(convert_cbvd5.py, convert_cvb.py) and the behavior training pipeline
(src/behavior/dataset.py). Contains bidirectional class↔ID mappings and
per-dataset source→canonical mapping dictionaries.

The label map is consumed as follows in code:

import json

with open(\"data/label_map.json\") as f:

label_map = json.load(f)

\# Map a CVB action_id to canonical class ID

cvb_name = cvb_action_id_to_name\[action_id\] \# e.g. \"Grazing\"

canonical =
label_map\[\"datasets\"\]\[\"CVB\"\]\[\"mapping\"\]\[cvb_name\] \#
\"Foraging\"

class_id = label_map\[\"class_to_id\"\]\[canonical\] \# 2

**7. Key Design Decisions and Justifications**

  ----------------------- -----------------------------------------------
       **Decision**                      **Justification**

   **Use official CBVD-5    Ensures comparability with Li et al. (2024)
         splits**          baseline results. Custom splits would make it
                            impossible to benchmark against the paper.

    **Use official CVB        Matches Zia et al. (2023) experimental
       80:20 split**      protocol. Treats val as test since no separate
                                       test set is provided.

   **Split by video, not  Prevents temporal data leakage. Adjacent frames
        by frame**          from the same video are highly correlated;
                           frame-level splits would inflate all metrics
                                           artificially.

          **Merge           Biologically, rumination is the behavior of
   Ruminating-standing +  interest. The postural component is captured by
    Ruminating-lying**    co-occurring Standing/Lying labels in CBVD-5's
                                        multi-label scheme.

    **Drop Hidden (not        Hidden is a visibility artifact, not a
        Walking)**        behavior. Walking was dropped for cross-dataset
                              consistency reasons, not because of low
                                            frequency.

    **Single class for        The detector's job is localization, not
        detection**           classification. Behavior recognition is
                            delegated entirely to VideoMAE operating on
                                         tracked tubelets.

   **Macro-F1 as primary  Class imbalance (Drinking = 2% of CBVD-5) makes
     behavior metric**      accuracy and micro-F1 misleading. Macro-F1
                           weights all classes equally and is robust to
                                            imbalance.
  ----------------------- -----------------------------------------------

**8. Summary**

Phase 0 established all foundational decisions required before model
training begins. The two selected datasets --- CBVD-5 (indoor, dairy,
sparse keyframe labels) and CVB (outdoor, beef, dense per-frame labels)
--- provide complementary coverage of real-world cattle farming
environments and enable rigorous cross-domain generalization
experiments. The 7-class unified taxonomy balances completeness with
cross-dataset comparability, and the frozen label map files
(labels.yaml, label_map.json) ensure that all pipeline stages operate on
identical, reproducible class definitions.

These decisions are **frozen and must not be modified** after this
point. Any change to class IDs, mappings, or dropped labels would
require retraining all models and regenerating all processed datasets
from scratch.

*Cattle Vision Framework --- Masters Thesis, Texas State University ---
2026*
