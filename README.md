# IntelliSummarizer

![Meeting Summarizer](https://github.com/user-attachments/assets/ec668f37-0f9b-4243-b437-17589cde48c6)

This project focuses on developing an intelligent model for automatic meeting summarization using fine-tuned large language models (LLMs). The goal is to generate high-quality summaries from meeting transcripts, leveraging models like BART, T5, and Pegasus. The research demonstrates the superior performance of a fine-tuned BART-Large model on the SAMSum dataset, evaluated using metrics such as ROUGE, BLEU, and METEOR.

## Model Access and Usage

- **Access the Model on Hugging Face**: [BART Samsum Model](https://huggingface.co/Arjun9/bart_samsum)
- **Try the Model**: [BART Samsum](https://arjun9-demo-summarization.hf.space)


## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Experimental Results](#experimental-results)
5. [Conclusion](#conclusion)
6. [Future Work](#future-work)


## Introduction

Meetings are crucial for organizational collaboration, decision-making, and information sharing. However, manually extracting critical insights from meeting transcripts has become laborious and ineffective due to the rise of remote connections and increasing interaction volumes. This project addresses this challenge by developing an intelligent approach using LLMs to automatically create high-quality summaries from meeting transcripts.

## Dataset

The SAMSum dataset, comprising around 16,000 chat-like conversations resembling real-life messenger interactions, is used for this research. Each conversation is accompanied by a human-written summary, making it an invaluable resource for dialogue summarization research in natural language processing.

## Methodology

### Data Preprocessing

The data preprocessing pipeline includes:
- Tokenization using the BART tokenizer from the Hugging Face Transformers library.
- Handling special characters and normalizing punctuation.
- Lowercasing all text for consistency.
- Truncation and padding of sequences to manage memory and batch processing efficiently.

### Model Selection

Three state-of-the-art language models were selected for the meeting summarization task:
1. **BART-Large**: A sequence-to-sequence model pre-trained using a denoising autoencoder objective.
2. **T5-Base**: A versatile model pre-trained on a unified text-to-text framework.
3. **Google Pegasus**: An optimized model tailored for abstractive text summarization.

### Fine-Tuning BART-Large

The BART-Large model was fine-tuned on the SAMSum dataset using the Seq2SeqTrainer object. The training process involved:
- Tokenizing the input conversations and summaries.
- Evaluating the model using ROUGE, BLEU, and METEOR metrics.
- Managing data batching and collation during training.

## Experimental Results

### Performance Metrics

The performance of the models was evaluated using the following metrics:
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**: Measures the overlap between the generated and reference summaries.
- **BLEU (Bilingual Evaluation Understudy)**: Evaluates the n-gram accuracy of the summary.
- **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**: Integrates synonyms, stemming, recall, and precision.

### Comparative Analysis

The fine-tuned BART-Large model demonstrated exceptional performance, outperforming T5-Base, Pegasus, and the standard BART-Large model across all ROUGE variants and METEOR scores. The results are summarized in the table below:

| Model                | rouge1 | rouge2 | rougeL | rougeLsum | BLEU | METEOR |
|----------------------|--------|--------|--------|-----------|------|--------|
| T5-Base             | 19.4735| 4.4057 | 15.7502| 15.7735   | 0    | 0.1760 |
| Pegasus             | 21.8362| 3.5137 | 17.2153| 17.1837   | 0    | 0.0284 |
| BART-Large          | 26.9533| 7.2353 | 21.2799| 21.2783   | 0    | 0.1946 |
| Fine-tuned BART-Large | 53.3294| 28.6009| 44.2008| 49.2031   | 0    | 0.4887 |

### Visualization

The performance of the models is visualized using bar charts, radar charts, and heatmaps, highlighting the superior performance of the fine-tuned BART-Large model.

## Conclusion

This research demonstrates the effectiveness of fine-tuning large language models for automatic meeting summarization. The fine-tuned BART-Large model shows significant improvements in generating high-quality summaries, capturing essential information from meeting transcripts. This technology has the potential to enhance productivity, communication, and decision-making in organizations.

## Future Work

Potential avenues for future enhancements include:
- Incorporating multimodal information such as visual cues from video recordings.
- Developing interactive and personalized summarization systems.
- Exploring domain-specific fine-tuning and transfer learning techniques.
- Investigating hybrid approaches that combine extractive and abstractive summarization techniques.
- Continuous advancements in language model architectures and training strategies.
