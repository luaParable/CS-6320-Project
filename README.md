# CS-6320-Project
# Automated Transcription of Air Traffic Control Tapes

## Overview
This project aims to create a prototype NLP system specialized for accurately transcribing and summarizing air traffic control communications. The intended benefit is to automate much of the time-consuming manual transcription process that follows aviation incidents and streamlines post-incident reviews.

## Overall Goal
- Reduce the burden on air traffic controllers by providing accurate, automated transcripts of ATC audio.
- Produce labeled transcripts that identify key entities (e.g., aircraft call signs, ATC positions).
- Generate concise summaries of incidents based on the transcribed audio.

## Scope
- Implement speech-to-text functionality using an open-source model.
- Fine-tune the model for specialized ATC terminology and noisy recordings.
- Produce a functioning prototype that can process publicly available ATC audio samples.
- Demonstrate entity labeling within transcripts.

## Team Member
- Paul Barela: Responsible for data sourcing, preprocessing, model training, evaluation, and prototype implementation.

## Data Sources
- Publicly available ATC audio archive: [LiveATC.net](https://www.liveatc.net)

## Implementation Approach

1. Initial Speech-to-Text Pipeline
   - Integrate Whisper to convert raw audio into text.
   - Establish a baseline transcription accuracy.

2. Fine-Tuning and Domain Adaptation
   - Collect a small labeled dataset of common ATC phrases and terminology.
   - Train the model on domain-specific vocabulary.

3. Entity Recognition
   - Use NLP techniques to identify important entities (aircraft identifiers, ATC positions).
   - Annotate transcripts with these labels for clarity.

4. Testing and Summarization
   - Evaluate transcription and entity labeling accuracy on unseen ATC samples.
   - Implement a summary generator that creates concise incident overviews.

## Project Steps

1. Requirement and Data Gathering
   - Identify data sources and create a plan for obtaining representative ATC samples.

2. Data Annotation / Test Writing
   - Create a labeled dataset of ATC communications.
   - Write test cases to measure transcription accuracy and entity-labeling performance.

3. Implementation / Model Training
   - Train or fine-tune Whisper using the annotated dataset.
   - Integrate NER (Named Entity Recognition) for entity tagging.
   - Develop a prototype application to handle audio input, run inference, and output transcripts.

4. Testing and Error Analysis
   - Evaluate performance on validation/test sets; measure WER (Word Error Rate) and label accuracy.
   - Diagnose errors and iterate on data labeling and training strategies.

5. Reporting and Presentation
   - Summarize the work, process, challenges, and lessons learned in a final report.
   - Demonstrate the prototype and present results.

## Future Improvements
- Additional training on real-world, noisy ATC audio to further boost accuracy.
- Integration with a front-end or chatbot interface for easier user interaction.
- More advanced summarization techniques to capture incident context precisely.


### Acknowledgments
- [LiveATC.net](https://www.liveatc.net) for publicly accessible ATC audio streams.
