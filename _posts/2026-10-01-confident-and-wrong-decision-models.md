---
layout: post
title: "Confident And Wrong: How Decision Models Work, And When To Trust Them"
image: "/posts/laya-title-img.svg"
tags: [Calibration, LLM, NLP, Machine Learning, Python]
description: "How a non-generative decision model like Laya answers a question in one forward pass, where it fits in an LLM pipeline, and why its confidence scores need calibrating before we act on them."
---

A new kind of model answers "which team should handle this ticket?" in about 33 milliseconds, without generating a single word, and returns a probability for every option. In this post we look at how that works, using the open-source [Laya](https://github.com/NandhaKishorM/laya) project as the example, and then at the harder question: when can we trust the probability it gives us?

# Table of contents

- [00. Overview](#overview-main)
    - [Context](#overview-context)
    - [What We Cover](#overview-actions)
    - [Key Takeaways](#overview-results)
- [01. Why Not Just Ask An LLM?](#why-not-llm)
- [02. How A Decision Model Works](#architecture)
- [03. Where It Sits In An LLM Pipeline](#pipeline)
- [04. What "Confidence" Has To Mean](#calibration-concept)
- [05. Training For Honesty: Proper Scoring Rules](#scoring-rules)
- [06. Case Study: Confident And Wrong](#case-study)
- [07. The Cheapest Fix: Temperature Scaling](#temperature)
- [08. One Threshold Does Not Fit All](#threshold)
- [09. A Practical Checklist](#checklist)
- [10. Growth & Next Steps](#growth-next-steps)

___

# Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Many steps in an AI pipeline are not writing tasks, they are decisions: which department gets this email, is this prompt an injection attempt, how urgent is this ticket. Today these are often handled by calling a large language model and parsing its reply, which is slow and expensive for what is really a classification.

In September 2026 the company TypeSafe launched Jev, a closed API built for exactly these decisions. Shortly afterwards an independent researcher, Nandakishor M, released Laya, an open-weight model for the same job. In its first two weeks the repository collected around 29,500 GitHub stars and close to 600 pull requests.

What makes Laya a good case study is that its repository publishes its own failures in detail. The headline promise of a decision model is a probability we can act on, and the project's own benchmark files show where that promise holds and where it does not.

<br>
<br>

### What We Cover <a name="overview-actions"></a>

The first half explains the mechanism: how a decision model turns a text and a list of options into probabilities in one pass, and where such a model fits alongside an LLM.

The second half is about *calibration*: whether a model that says "90% sure" is right 90% of the time. We cover what calibration means, how proper scoring rules try to train for it, and how temperature scaling repairs it afterwards.

**A note on the numbers:** I have not run Laya myself for this post. Every figure quoted comes from the result files, benchmark report and issues published in the project's repository, and each chart names its source file. Figures for Jev are third-party measurements quoted by that repository.

<br>
<br>

### Key Takeaways <a name="overview-results"></a>

* A decision model scores a fixed list of options in a single forward pass. Nothing is generated, so there is no output to parse.
* It works best as a fast first layer for routing, guardrails and triage, with an LLM or a person behind it for the unsure cases.
* Training with a proper scoring rule rewards honest probabilities, but only on data that looks like the training data.
* As shipped, the calibration error of Laya's English checkpoint averages **0.466** across 49 benchmark suites. Refitting the temperature on held-out data brings it to **0.081**.
* A confidence threshold is a policy we fit on our own data. It does not carry over to a new language or to a different number of options.

___

# Why Not Just Ask An LLM?  <a name="why-not-llm"></a>

A generative LLM produces its answer one token at a time. Each token needs a full pass through the network, and each new token is fed back in to produce the next. A short JSON reply such as `{"dept": "billing"}` already costs several passes, and then our code has to parse the text and hope it is valid.

The LLM can also write a confidence value if we ask for one, but that number is generated text like any other. Nothing ties it to how often the model is right.

A decision model takes a different route. We give it the text and the list of allowed answers, and it returns one probability per answer from a single pass.

<br>

![Schematic comparing a generative LLM, which writes a JSON answer token by token, with a decision model, which returns one probability per option in a single forward pass](/img/posts/laya-llm-vs-decision-model.svg "Generative LLM vs decision model")

<br>

On a Tesla T4 GPU, the Laya repository measures 33 to 40 ms for one question, and 7 to 16 ms per question when ten are batched. Independent benchmarks quoted there put Jev's hosted API at 236 to 276 ms per question.

___

# How A Decision Model Works  <a name="architecture"></a>

Laya supports three question types:

| **Type** | **Returns** | **Typical use** |
|---|---|---|
| `choice` | One label from a list, with a probability per label | Department routing, intent detection |
| `score` | A level on an ordered scale | Urgency, frustration, severity |
| `noul` | A yes/no probability | Phishing, prompt injection, churn risk |

<br>

In code, a request looks like this (from the project's quickstart):

```python
from laya import Router

router = Router()

state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."

questions = {
    "department": {"type": "choice",
                   "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul",
                   "instructions": "Does the user threaten to cancel or leave?"},
}

result = router.predict(state, questions)
result["answers"]["department"]["choice"]   # billing
result["answers"]["churn_risk"]["noul"]     # probability the answer is yes
```

<br>

The mechanism behind it is simple once the input layout is clear. For each question, the model builds one sequence: the question, then every option preceded by a `[MASK]` token, then the text to judge.

<br>

![Architecture diagram: the input sequence holds the question, one MASK token per option and the text; an encoder and a decision head process it; the hidden state at each MASK is scored and a temperature-scaled softmax produces the probabilities](/img/posts/laya-architecture.svg "Laya architecture")

<br>

Three design choices are worth noting:

* **The encoder is bidirectional.** Laya uses ModernBERT-large, a model from the BERT family in which every token can attend to every other token. Each option therefore "reads" the whole text, and the text "reads" the options.
* **Each `[MASK]` is a slot for one option.** After the encoder and a small two-layer decision head, the model keeps only the vectors at the `[MASK]` positions and passes each through a small network that outputs a single score, called a logit.
* **A softmax turns scores into probabilities.** The logits are divided by a *temperature* `T` and normalised so they sum to 1. We come back to `T` later, because it controls how confident the model sounds.

Several questions about the same text are stacked into one batch and answered in the same pass. The whole model has 421 million parameters, small enough for a single modest GPU or a slower CPU.

The design also has a cost. All options share a fixed token budget (192 or 256 tokens by default), so a question with 77 options leaves three or four tokens per option. On the 77-label Banking77 benchmark Laya scores 0.425 against Jev's published 0.870, and the repository recommends keeping choice questions under about 20 options.

___

# Where It Sits In An LLM Pipeline  <a name="pipeline"></a>

A decision model does not replace the LLM. It sits in front of it and answers the frequent, cheap questions, so that the expensive model, or a person, only sees the cases that need it.

<br>

![Pipeline diagram: a request passes a language check, a decision model answers guardrail, routing and triage questions in one pass, and a confidence gate sends the result to block, act automatically or escalate](/img/posts/laya-pipeline.svg "Decision model in an LLM pipeline")

<br>

The three common roles are:

* **Guardrail:** is this prompt a jailbreak or an injection attempt? Checked before any LLM sees it.
* **Routing:** which team, tool or model should handle this request? Simple requests go to a small model, hard ones to a frontier model.
* **Triage:** how urgent or how risky is this item? Used to order a queue.

Everything to the right of the model in this diagram depends on one box: the confidence gate. If the confidence is trustworthy, we automate the confident cases and escalate the rest. If it is not, the gate lets wrong answers through with a clean conscience. That is the subject of the rest of this post.

The repository's own adoption guide is cautious on this point. It recommends running the model in "shadow" mode first, logging its decisions next to the existing process, and promoting only a small, reversible slice once the measured accuracy supports it.

___

# What "Confidence" Has To Mean  <a name="calibration-concept"></a>

A model is **calibrated** when its stated confidence matches how often it is right. Of all the answers it gives at 80% confidence, about 80% should be correct.

Calibration is separate from accuracy. A model can be accurate and badly calibrated (right 80% of the time but always claiming 99%), or mediocre and well calibrated (right 60% of the time and saying so). For automation, the second is often more useful, because we can tell which answers to hand to a person.

The standard measure is **Expected Calibration Error (ECE)**:

1. Sort the predictions into bins by confidence (Laya's benchmarks use 15 bins).
2. In each bin, compare the average confidence with the actual accuracy.
3. Average the gaps, weighted by how many predictions fall in each bin.

An ECE of 0 is perfect. An ECE of 0.40 means that, on average, stated confidence is 40 percentage points away from reality.

___

# Training For Honesty: Proper Scoring Rules  <a name="scoring-rules"></a>

How do we train a model to report honest probabilities? The reward has to make honesty the best strategy.

Suppose an event is truly 70% likely and the model must report a probability `q`. If we reward the model in proportion to the probability it placed on what actually happened, its best move is to report 100%: the reward keeps rising as `q` rises. That reward teaches overconfidence.

A **strictly proper scoring rule** is a reward whose average is highest only when the reported probability equals the true one. The log score and the spherical score are two examples.

<br>

![Three charts of expected reward against reported probability when the true probability is 70 percent: an accuracy-style reward peaks at 100 percent, while the log score and the spherical score peak at 70 percent](/img/posts/laya-proper-scoring-rules.svg "Proper scoring rules")

<br>

Laya's training method, which its author calls RLCD (Reinforcement Learning for Calibrated Decisions), uses a sum of such rules as the reward:

```
reward = log score + 0.5 × spherical score − ranked probability score
```

The ranked probability score applies only to `score` questions. It penalises a guess that is far from the true level more than a near miss, so answering "level 2" when the truth is "level 3" costs less than answering "level 0".

This is a sound foundation, with one limit that matters for everything below. A proper scoring rule makes honesty optimal *on the data the model is trained on*. It promises nothing about text that looks different from that data.

___

# Case Study: Confident And Wrong  <a name="case-study"></a>

Laya ships an English checkpoint and a multilingual one, plus a third fine-tuned for one benchmark. The repository tested the first two on the same 20-option intent question in 51 languages, 100 cases per language. The chart below plots, for each language, how confident the model was on average against how often it was right.

<br>

![Scatter plot of mean confidence against accuracy for 51 languages and two Laya checkpoints; every point lies below the diagonal of perfect calibration](/img/posts/laya-confidence-vs-accuracy.svg "Confidence vs accuracy across 51 languages")

<br>

Every point sits below the diagonal, so both checkpoints claim more than they deliver. The English checkpoint is the striking one:

* In English it is 82% accurate at 96% mean confidence.
* Across all 51 languages its accuracy averages 0.23 while its calibration error averages 0.57.
* In Khmer it gets **0 of 100** cases right at 71% mean confidence (95% before a safety clamp added in a later release).

The last point is the important one for pipeline design. The model's confidence does not fall when it meets a language it cannot read, so a confidence gate cannot catch the failure. This is why Laya runs a language check *before* the model: a cheap script-and-language detector sends non-English text to the multilingual checkpoint. The safeguard had to be placed outside the model, because the model's own signal gave no warning.

___

# The Cheapest Fix: Temperature Scaling  <a name="temperature"></a>

Recall that the model computes `softmax(z / T)`. The temperature `T` rescales the logits before they become probabilities:

* `T` below 1 sharpens the distribution, so the model sounds more certain.
* `T` above 1 flattens it, so the model sounds less certain.
* The order of the options never changes, so accuracy is untouched.

<br>

![Three bar charts of the same five logits at temperatures 0.5, 1 and 2: the top option's probability falls from 0.78 to 0.52 to 0.35 while the ranking stays the same](/img/posts/laya-temperature-scaling.svg "Effect of temperature on softmax probabilities")

<br>

**Temperature scaling** means choosing `T` after training, on held-out data, so that stated confidence matches measured accuracy. It is a single number for each kind of question, it takes seconds to fit, and it cannot change accuracy, because the ranking of the options stays the same.

Laya's repository ran this repair on 49 benchmark suites for both checkpoints:

<br>

![Strip plot of calibration error across 49 benchmark suites: as shipped the means are 0.466 and 0.314, after refitting the temperature they are 0.081 and 0.106](/img/posts/laya-ece-shipped-vs-refit.svg "Calibration error as shipped vs after temperature refit")

<br>

Mean calibration error falls from 0.466 to 0.081 on the English checkpoint and from 0.314 to 0.106 on the multilingual one.

The same data shows that a badly fitted temperature can do harm. For questions with 11 or more options, the English checkpoint ships with `T = 0.10`, which sharpens every answer roughly tenfold. A repository issue measured its effect on one 20-option question:

| **Temperature for 11+ options** | **Probability of top answer** | **Options above 1%** |
|---|---|---|
| 0.10 (as shipped) | 1.000 | 1 of 20 |
| 0.5 (after the safety clamp) | 0.951 | 3 of 20 |
| 1.0 (no scaling) | 0.566 | 16 of 20 |

<br>

On the English intent benchmark, this shipped temperature gives an ECE of 0.214, while applying no temperature at all gives 0.126. On that benchmark the "calibrated" setting was worse than the raw model. A fitted temperature is only as good as the match between the data it was fitted on and the data it meets.

___

# One Threshold Does Not Fit All  <a name="threshold"></a>

The usual way to use confidence is a gate: act automatically above a threshold, escalate below it. An early version of Laya's README suggested `confidence >= 0.85`.

A contributor tested that gate on the same 12 texts, asked once as a 3-option question and once as a 20-option question. The sample is tiny, so this is a demonstration, not a benchmark, but the pattern is clear.

<br>

![Grouped column chart of gate precision at four confidence thresholds: with 3 options precision is 9 of 10, 7 of 7 and 3 of 3; with 20 options it is 7 of 12, 6 of 11, 5 of 9 and 5 of 7](/img/posts/laya-threshold-vs-options.svg "Gate precision by threshold and option count")

<br>

With 3 options the gate works as intended: everything it passes at 0.70 or above is correct. With 20 options, the gate at 0.85 passes 9 of 12 answers and only 5 are right, which is slightly worse than the model's ungated 7 of 12. The average confidence is 0.915 for the right answers and 0.906 for the wrong ones, so a threshold has almost nothing to separate.

The README now shows a placeholder `THRESHOLD` and states that it must be fitted on the reader's own data, at the option count the workload uses.

___

# A Practical Checklist  <a name="checklist"></a>

For anyone putting a decision model, or any classifier with a confidence score, in front of an automated action:

* **Measure calibration on your own data.** Plot confidence against accuracy, as in the 51-language chart above, before trusting a threshold.
* **Refit the temperature on held-out data** that matches production: same language, same question, same number of options.
* **Choose the threshold from measured precision,** at the share of traffic you want to automate. Do not copy a number from documentation.
* **Guard the inputs separately.** Confidence will not flag text the model cannot read, so check language and format before the model runs.
* **Keep options few.** Under about 20 per question for this architecture.
* **Start in shadow mode.** Log decisions next to the current process, compare, and automate a small reversible slice first.
* **Re-measure when anything changes:** the checkpoint, the question wording, the option list or the traffic mix.

___

# Growth & Next Steps  <a name="growth-next-steps"></a>

This post relies on the project's published numbers. The natural next step is to run the experiment myself: take a public support-ticket dataset, measure Laya's accuracy and calibration before and after a temperature refit, and work out what share of tickets a gate could safely automate.

It would also be worth comparing against two baselines on the same data: a small LLM asked for a label and a confidence, and a plain fine-tuned classifier. The repository's own results suggest that fine-tuning is where most of the value lies. On its typed-decisions benchmark, the base checkpoint scores 0.362, below the 0.461 obtained by always guessing the most common answer, while the checkpoint fine-tuned on that benchmark's training data reaches 0.766.

<br>

**Sources**

* [Laya repository](https://github.com/NandhaKishorM/laya): README, `BENCHMARKS.md`, `docs/staged-adoption.md`, `laya/common.py` and the result files under `research/results/`
* [Issue #394](https://github.com/NandhaKishorM/laya/issues/394): a confidence threshold does not transfer across option counts
* [The author's write-up](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me) on the architecture and the RLCD training method
