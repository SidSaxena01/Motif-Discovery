# Title Page

- **Title:** Automatic Motif Detection in the Star Wars Soundtrack
- **Authors:** [Siddharth Saxena, Andreas Papaeracleous, Fernando Garcia de la Cruz]
- **Affiliation:** [Universitat Pompeu Fabra]
- **Date:** [25 March 2025]

---

# Abstract

- **Summary (150–300 words)** that briefly explains:
  - The problem tackled (automatic motif detection/identification)
  - The dataset used (Star Wars dataset)
  - The methods employed (melody extraction with CREPE and Melodia; motif and thematic information extraction from XML via music21)
  - Key approaches (using stumpy matrix profile and DTW for motif detection)
  - Preliminary findings & significance
- **Note:** Keep the conclusions tentative since you’re still fine-tuning hyper-parameters and exploring recurrent motif extraction.

This project addresses the challenge of automatic motif detection in musical recordings using the Star Wars dataset. We extract melody information from the audio using both CREPE and Melodia, and extract melodic themes from XML files via music21. For motif detection, we employ both the stumpy matrix profile method and DTW, finding that stumpy offers semi-accurate results with notably faster computation, while DTW, though effective, is significantly slower.
Additionally, we investigate recurrent motif extraction across different audio tracks using DTW.
A notable challenge has been working with melody extraction on orchestral music, which is still an open challenge to be solved.
Overall, our work demonstrates a promising hybrid approach that combines melody extraction with pattern detection techniques to identify musical motifs in audio recordings.

or

This project addresses the challenge of automatically detecting recurring musical motifs in film soundtracks by focusing on the Star Wars dataset. Our approach leverages a dual strategy: first, extracting melody information from raw audio using two contrasting methods – the CREPE deep learning model and the signal processing-based Melodia algorithm – and second, aligning this extracted melodic information with symbolic representations obtained from XML files via music21. For motif detection, we experiment with two methods: the stumpy matrix profile approach, which identifies candidate motifs through fast computations, and Dynamic Time Warping (DTW), known for its precise albeit computationally heavier alignments. Early findings indicate that while stumpy offers faster results with acceptable accuracy, DTW provides finer temporal alignment at the cost of increased computation. Our pipeline also explores cross-track motif comparison to find recurrent patterns across different audio recordings. This hybrid approach demonstrates promise for motif and leitmotif extraction in orchestral scores, though significant challenges remain in data synchronization and parameter tuning. These preliminary results underscore the potential of merging deep-learning and classical signal processing methods for musical pattern discovery in complex soundtracks.

---

# 1. Introduction

## 1.1 Background and Motivation
- Present the broader context of Music Information Retrieval and the challenge of motif detection.
- Discuss the importance of detecting thematic materials (motifs, leitmotifs) in media soundtracks.
- Define key terminology (e.g., motif, leitmotif, melody extraction, etc.).

In the field of Music Information Retrieval (MIR), automatic motif detection is a challenging task that aims to uncover repeated thematic material in musical recordings. Motifs and leitmotifs are short, recurring patterns that serve as thematic identifiers in media such as film and video game soundtracks. Detecting these elements can enable richer musicological analysis and multimedia retrieval applications. However, variable renditions of motifs—such as variations in tempo, orchestration, or rhythmic alterations—make automated detection particularly difficult. This project addresses this gap by developing an end-to-end system specifically tailored for the Star Wars soundtrack, leveraging state-of-the-art methods from both traditional signal processing and deep learning.


## 1.2 Project Objectives
- Clearly lay out the goals of the project:
  - To develop an end-to-end pipeline for motif detection with automatic extraction of musical content.
  - To compare traditional methods (signal processing-based like stumpy/matrix profile and DTW) on the Star Wars dataset.
  - To evaluate and refine the pipeline's performance on known motif occurrences.
- Briefly state the approaches used and the challenges being addressed (like hyper-parameter tuning, cross-track comparisons).


The primary objectives of this project are to:
- Develop an automated pipeline for motif detection that integrates both audio-based melodic feature extraction and symbolic thematic alignment.
- Compare and evaluate traditional algorithmic approaches (matrix profile via stumpy and DTW) for motif detection on the Star Wars dataset.
- Bridge the gap between raw audio analysis and symbolic music representation to extract consistent motif candidates.
- Address challenges related to data synchronization, computational efficiency, and detecting motifs across different audio tracks.


## 1.3 Overview of the Report
This report is organized as follows. Section 2 reviews related work in motif detection and discusses several publicly available datasets and annotation challenges. Section 3 details our methodology, including data preprocessing, melody extraction using CREPE and Melodia, and the motif detection methods employed. Section 4 provides our experimental setup and pipeline design. Section 5 presents preliminary results, followed by a discussion of challenges and open questions in Section 6. Finally, Section 7 outlines priorities for future work, and Section 8 concludes the report.

---

# 2. Related Work

## 2.1 Literature Review
- Summarize previous work on motif detection using matrix profiles, DTW, and deep-learning methods (e.g., Krause et al., Nuttall et al., Ganguli et al.).
- Place your project in context with how your work draws from and builds upon these methods.

Previous research in motif detection has spanned both signal processing and deep learning approaches. The matrix profile method has been used successfully in works such as Nuttall et al. (2023) for motif discovery in Carnatic music, demonstrating the utility of time-series analysis in identifying recurring segments. Alternative approaches, such as replacing DTW with string-matching techniques (Ganguli et al., 2017), highlight how symbolic abstraction can reduce computational complexity while still capturing musical nuance. Additionally, parameter-free methods (Hao et al. 2013) have opened research avenues in generalized audio pattern discovery across domains such as animal vocalizations and industrial sound processes. More recently, deep-learning strategies, as presented by Krause et al. (2021), have focused on detecting leitmotif activities in opera recordings. Our work builds on these insights, applying them to the domain of film soundtracks and addressing issues of data synchronization between symbolic and audio representations.


## 2.2 Datasets in Music Information Retrieval (MIR)
- Provide an overview of the Star Wars thematic corpus and other datasets mentioned in the literature.
- Discuss open challenges in dataset annotation and motif occurrence identification.

The Star Wars thematic corpus contains multiple themes in symbolic formats XML, humdrum, as well as some youtube links with time-stamps of the motifs in action (at most one per movie).
In the wider MIR community, datasets with annotated motifs are scarce, particularly in orchestral and film music. Our work aims not only to validate motif detection techniques on this challenging dataset but also to lay groundwork for improved annotation strategies in future research.

---

# 3. Methodology

## 3.1 Data Description and Preprocessing

### Thematic Information Extraction via music21
- Explain how music21 was used to extract thematic (symbolic) information from XML files.
- Describe the transformation of symbolic data into arrays (mention interpolation strategy to match beats per minute and analysis frames).

We use the music21 library to parse XML files and extract thematic elements (motifs, phrases, etc.) into arrays of frequency values. This transformation involves converting note sequences and their durations into time-aligned data, thereby enabling a comparison between extracted melodic features from the raw audio and the annotated motifs.


## 3.2 Audio-based Melody Extraction

### CREPE:
- Explain the use of the CREPE deep learning model in extracting pitch information.
- Describe any preprocessing steps (e.g., segmentation, normalization).

The CREPE deep learning model is employed for a direct pitch estimation from the audio. CREPE is known for its robustness in estimating pitch from complex audio environments. Preprocessing steps include normalization of audio levels, and conversion to a sample rate of 16 kHz to ensure consistency across recordings.

### Melodia:
- Describe the Melodia algorithm for predominant melody extraction.
- Explain its parameters and any differences observed compared to CREPE.

In contrast, the Melodia algorithm applies a signal processing-based approach to extract the predominant melody from the audio. Although parameter tuning is crucial to its performance on orchestral soundtracks, we did not explore different configurations and kept to the defaults defined in Essentia. Compared to CREPE, Melodia can sometimes better leverage signal structure, although it is more sensitive to noise and overlapping sounds.

### Comparative Discussion:
- Briefly discuss the trade-offs between the two methods with regard to accuracy, computational load, etc.

Initial comparisons reveal trade-offs between CREPE and Melodia: CREPE tends to be more robust in polyphonic contexts due to its deep-learning background, while Melodia often provides clearer pitch contours in relatively cleaner segments thanks to the viterbi encoding. Neither approach's output led to a significant improvement in motif detection, so we opt to use Melodia as it is computationally more efficient.

## 3.3 Motif Detection Techniques

### Stumpy (Matrix Profile Approach)
- Describe the stumpy library’s role in motif discovery.
- Explain how the matrix profile method is applied to identify segments in the extracted melody that are candidate motifs.
- Include discussion on hyper-parameter choices (window length, exclusion zones) and settings experimented.

The stumpy library implements the matrix profile, a tool for rapid discovery of similar sub-sequences in time series data. Here, the matrix profile is computed on the extracted melody (the pitch contour). This method quickly identifies candidate motifs and provides an efficient baseline for motif detection.

### Dynamic Time Warping (DTW)
- Provide an overview of DTW for aligning time-series data.
- Explain its application for motif detection/comparison and why it is computationally slower.

DTW is employed to precisely align time series sequences, allowing for consideration of tempo variations and minor interpretative changes in motif presentations. Although DTW produces more accurate temporal alignments compared to the matrix profile, its higher computational complexity makes it less practical for large-scale real-time applications. DTW is used primarily for validation and refinement of motif candidates.

## 3.4 Cross-Track Motif Comparison
- Describe the methodology for comparing motifs across different audio tracks.
- Mention preliminary explorations regarding recurrent motif extraction when comparing two audio tracks.
- State that while no conclusions are drawn yet, this remains a significant aspect of the ongoing work.

Motif detection across different tracks is performed by using `stumpy.mass`, which analyses pairs of pitch sequences and looks for recurrent patterns.

---

# 4. Experimental Setup

## 4.1 Pipeline Design
- Provide a flowchart or diagram summarizing your end-to-end pipeline:
  - **Input:** Star Wars audio and XML files
  - **Stages:** Thematic extraction → Melody extraction → Data transformation (synchronization via BPM/interpolation) → Motif detection via stumpy and DTW → Cross-track analysis.
- Brief notes on the design rationale (why certain steps or methods were prioritized).

The pipeline is structured as follows:
- **Input:** Star Wars audio recordings (WAV) and corresponding XML files.
- **Stage 1:** Thematic Extraction – Use music21 to extract symbolic motif data from XML.
- **Stage 2:** Melody Extraction – Process audio via CREPE and Melodia independently to create pitch contour arrays.
- **Stage 3:** Data Transformation – Align the extracted melody with symbolic data using BPM normalization and interpolation strategies.
- **Stage 4:** Motif Detection – Apply the stumpy matrix profile method and refine motif candidates with DTW.
- **Stage 5:** Cross-Track Analysis – Compare motif candidates across different audio tracks using pairwise DTW alignment.
A flowchart detailing these steps is maintained in the project documentation to guide further refinements.

---

# 5. Experimental Results and Analysis (Preliminary)

## 5.1 Performance of Melody Extraction
- Compare results from CREPE and Melodia:
  - Present qualitative or quantitative observations (e.g., clarity of the extracted melody lines, alignment with the symbolic arrays).

Initial comparisons reveal trade-offs between CREPE and Melodia: CREPE tends to be more robust in polyphonic contexts due to its deep-learning background, while Melodia often provides clearer pitch contours in relatively cleaner segments thanks to the viterbi encoding. Neither approach's output led to a significant improvement in motif detection, so we opt to use Melodia as it is computationally more efficient.


## 5.2 Motif Detection Results
- Report on the detection of known motifs in tracks where they were expected.
- Provide preliminary findings regarding:
  - Accuracy of stumpy in detecting motifs and sensitivity to hyper-parameter variations.
  - Comparative results with DTW (noting the computational cost and performance differences).
- Illustrate findings with visual examples:
  - Insert figures like plots of the matrix profile, DTW alignment graphs, and overlaid motif candidates on melody contours.



## 5.3 Cross-Track Motif Extraction
- Describe initial experiments on motif similarity across different tracks.
- Detail any patterns or potential recurrent motifs discovered, along with current limitations.

## 5.4 Discussion of Challenges
- Discuss any challenges encountered during:
  - Data synchronization between symbolic musical information and audio.
  - High computational cost when using DTW.
  - Sensitivity of the motif detection algorithms to the chosen parameters.
- Note areas where additional experimentation or computational improvements are planned.

Challenges encountered include:
- Synchronization between the noisy audio-derived melody and the cleaner symbolic representations from XML.
- High computational cost when applying DTW over larger datasets, particularly when multiple candidate motifs need validation.
- Sensitivity of the stumpy matrix profile to parameter changes; small adjustments result in notable differences in motif segmentation.
Continued experimentation and potential algorithm refinements (such as variable window lengths for motif detection) are planned to address these issues.


---

# 6. Discussion

## 6.1 Interpretation of Preliminary Results
- How do the results validate the approach of combining audio-based extraction with symbolic thematic alignment?
- Are there discrepancies between the annotated motifs (from music21 data) and the detected motifs?

## 6.2 Comparison of Methods
- Provide insights on the strengths and weaknesses of using stumpy versus DTW.
- Discuss the balance between computational efficiency and detection accuracy.

## 6.3 Limitations and Open Questions
- Describe any limitations in the current pipeline (e.g., scalability, dependence on manual tuning, challenges in cross-track analysis).
- Summarize open questions that the ongoing experiments aim to resolve.

---

# 7. Future Work

## 7.1 Pipeline Refinement
- Plans to optimize the hyper-parameters further and possibly incorporate automated parameter search.

## 7.2 Enhanced Cross-Track Analysis
- Outline future experiments to better extract recurrent motifs across different audio recordings.

## 7.3 Potential for Deep Learning Integration
- Briefly state intentions to explore deep-learning methods to overcome limitations in traditional DTW and matrix profile approaches, if relevant.

## 7.4 Expanding the Dataset or Annotations
- State intentions regarding possible manual annotations or augmentation of the dataset for improved grounding of motif detection.

---

# 8. Conclusion

- Recap the work done so far and its significance.
- Reiterate the potential impact of your proposed pipeline for automatic motif detection.
- Remind the reader that final conclusions will be drawn as experiments finalize, and stress the contribution of early-stage findings to the field.

This report details an end-to-end pipeline for automatic motif detection in the Star Wars soundtrack. By merging traditional melodic extraction techniques with state-of-the-art methods such as the matrix profile (stumpy) and DTW, preliminary results demonstrate the feasibility of the approach despite several challenges. Our work contributes a promising hybrid method that bridges audio analysis with symbolic alignment, opening avenues for enhanced musical retrieval and media analysis. Final conclusions will be drawn as further experiments refine the pipeline and overcome current limitations.

---

# 9. References
- Nuttall, T., Plaja-Roglans, G., Pearson, L., & Serra, X. (2023). The Matrix Profile for Motif Discovery in Audio—An Example Application in Carnatic Music. In M. Aramaki et al. (Eds.), Music in the AI Era (Vol. 13770, pp. 228–237). Springer. https://doi.org/10.1007/978-3-031-35382-6_18

- Ganguli, K. K., Lele, A., Pinjani, S., Rao, P., Srinivasamurthy, A., & Gulati, S. (2017). Melodic shape stylization for robust and efficient motif detection in Hindustani vocal music. 2017 Twenty-Third National Conference on Communications (NCC), 1–6. https://doi.org/10.1109/NCC.2017.8077055

- Hao, Y., Shokoohi-Yekta, M., Papageorgiou, G., & Keogh, E. (2013). Parameter-Free Audio Motif Discovery in Large Data Archives. 2013 IEEE 13th International Conference on Data Mining, 261-270. doi:10.1109/ICDM.2013.30

- Krause, M., Müller, M., & Weiß, C. (2021). Towards Leitmotif Activity Detection in Opera Recordings. Transactions of the International Society for Music Information Retrieval, 4(1), 127–140. https://doi.org/10.5334/tismir.116

- Additional documentation and repository links for tools such as CREPE, Melodia, music21, stumpy, and DTW libraries.

---

# 10. Appendices

- **Appendix A:** Additional Figures/Visualizations of Motif Detection Results
- **Appendix B:** Code Snippets or Pseudocode of Key Pipeline Steps
```
