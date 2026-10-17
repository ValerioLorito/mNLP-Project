# Dense Retrieval Fine-Tuning & Knowledge-Augmented RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Wikidata](https://img.shields.io/badge/API-Wikidata-green)

An end-to-end Natural Language Processing project exploring the fine-tuning of dense retrievers for semantic search and leveraging knowledge graphs (Wikidata) to augment Retrieval-Augmented Generation (RAG) systems for Question Answering tasks.

---

## Project Overview

This repository contains two main modules developed for the **Multilingual Natural Language Processing (MNLP)** course at Sapienza University of Rome:

1. **Module 1: Semantic Search Fine-Tuning & Dense Retrieval**
   * Empirical investigation comparing **Multiple Negatives Ranking Loss (MNR)** vs. **Contrastive Sentence Pairs Loss (CSP)**.
   * Fine-tuning monolingual (`all-MiniLM-L6-v2`, `distilbert-base-uncased`) and multilingual (`F2LLM-v2-80M`) sentence embedding models.
   * Hard negative mining using GIST (`avsolatorio/GIST-small-Embedding-v0`) combined with dynamic in-batch negative sampling.

2. **Module 2: Wikidata-Augmented RAG Pipeline & Generative QA**
   * Implementation of a RAG architecture contrasting **Seq2Seq** (`flan-t5-large`) and **Causal** (`LLaMA-3.2-1B-Instruct`) generative models.
   * Custom **Wikidata Information Retrieval Engine** that fetches, filters, and translates entity properties into structured text (`Document 4`).
   * **Dynamic Ground Truth Expansion (Alias Resolution)**: Automated alias extraction via Wikidata API to mitigate strict evaluation bias in Exact Match (EM) and subEM metrics.
   * **LLM-as-a-Judge Evaluation**: Automated evaluation with `Mistral-7B-Instruct-v0.3` (zero-temperature, reasoning-first setup) validated against human annotation.

---

## 📂 Project Structure

### Key Methodology
1. Hard Negative Mining & Loss Strategies
* Contrastive Pairs (CSP): Formatted as (anchor, positive/negative, label) triples combining GIST-ranked hard negatives with random easy negatives.
* Multiple Negatives Ranking (MNR): Formatted as n-tuples. Exploits in-batch negatives (batch size = 8) as dynamic easy negatives, preventing spatial collapse in the vector space.

3. Wikidata Knowledge Engine (wikidata_utils.py)
Context Enrichment Workflow (Algorithms 1–3):
* Fetches entity labels, descriptions, and aliases.
* Restricts extracted claims to a maximum of 10 properties, prioritizing a curated set of 20 "Gold Properties" (e.g., location, point in time, instance of).
* Automatically resolves Wikidata IDs (e.g., Q220) into human-readable English text.

Ground Truth Expansion (Algorithm 4):
* Matches short answers against entity claims to retrieve true Wikidata IDs.
* Dynamically populates acceptable answer aliases (e.g., expanding "United States" to {"usa", "u.s.", "america"}), ensuring fair scoring for generative outputs.

### Performance Highlights
##### Phase 1: Semantic Search Results (Test Set)
| Model	Strategy | Hit@1 | Hit@3 | Hit@5 |MRR@10
|----------------|------|------|------|------|
| all-MiniLM-L6-v2 (CSP) | 0.6010 |	0.8185 | 0.8910 |	0.7212 |
| all-MiniLM-L6-v2 (MNR) | 0.6105 | 0.8450 | 0.9140 | 0.7386 |
| F2LLM-v2-80M (CSP) | 0.5615 |	0.8005 | 0.8745 |	0.6937 |
| F2LLM-v2-80M (MNR) | 0.5905 | 0.8170 | 0.8970 | 0.7205 |

##### Phase 2: Generative QA & RAG Pipeline (2000-Query Test Set)
|Model	Pipeline |	Standard EM |	Wikidata EM | Standard subEM | Wikidata subEM | METEOR (WD)
|----------------|------|------|------|------|------|
|FLAN-T5-large (Baseline) |	0.0245 |	0.0280 |	0.0350 |	0.0495 |	0.0717 |
|FLAN-T5-large (RAG) |	0.3475 |	0.3785 |	0.4505 |	0.5200 |	0.4873 |
|FLAN-T5-large (Oracle) |	0.4490 |	0.4250 |	0.5435 |	0.6005 |	0.5547 |
|LLaMA-3.2-1B	(RAG) |	0.0595 |	0.0670 |	0.3525 | 0.4410 |	0.2929 |

**Human vs. LLM-as-a-Judge Agreement:** Mistral-7B-v0.3 achieved an average correctness score of 0.7480 vs. 0.7440 by human annotators, yielding a Cohen's Kappa of 0.6728.

### Colab Notebooks
Both experiments are fully reproducible on the two notebooks in the repository:
Phase 1: mnlp2026_hw1.ipynb
Phase 2: mnlp2026_hw2.ipynb
