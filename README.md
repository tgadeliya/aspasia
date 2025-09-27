# Aspasia

Small multi-agent framework for running debate protocols using Inspect framework

## Description

This framework built on in Inspect framework and implements protocols, agents and used for . As the initial point, paper "Debating with More Persuasive LLMs Leads to More Truthful Answers" is used

## Relevant papers
 - Debating with More Persuasive LLMs Leads to More Truthful Answers
 - An alignment safety case sketch based on debate
 - AI safety via debate 
 - Debate Helps Supervise Unreliable Experts
 - Scalable AI Safety via Doubly-Efficient Debate

## Protocols
<p align="center">
  <img src="img/debate_protocols_from_paper.png" alt="debate protocols from the paper"/>
  <br>
  <em>Protocols from "Debating with More Persuasive LLMs Leads to More Truthful Answers"</em>
</p>

### Consultancy
Given article and question (based on the article) and two answering options, consulter are asked to choose the best answer basedo nt he article. Before final decision, judge could provide multiple question-answer turns to clarify consultant answer and argumentation. Judge doesn't have access to article, but consulter can add citations from the article to support its position.

### Debate
Every debater assigned with specific option to debate. In QuALITY, every question assume two options, hence one agent assigned with A and other with B. They conduct N turns of debate and after that judge decides what option to choose. Judge doesn't have access to article, but debaters can add citations from the article to support its position.

### Interactive Debate
Similar to debate, but judge can interact with debaters after every turn, similar to Consultancy


### Additional options

#### Human judge
In every protocol LLM judge could be changed to real person.

#### Symmetric 
Judge has access to all information as debater/counsulter does, except consulter/debater thinking (text in <thinking> </thinking> tags)

## Usage
### Dataset

 TODO: Add from scetch blabla paper description of Quality dataset.

1. Download  `QuALITY.v1.0.1` dataset from the original repo. By default, (see Makefile) files download into repo root directory `data`.
```bash
make download-data
```


2. Load pre-processed dataset split in  `MemoryDataset` compatible with Inspect Task. Pre-processing is done according to instruction from `Debating with More Persuasive LLMs Leads to More Truthful Answers` appendix.
Available splits: `train`, `dev`, `test`

```python
    from pathlib import Path
    from aspasia.datasets import QuALITY
    dataset = QuALITY(Path("PATH_TO_DATA_DIR")).prepare_datasets("dev")
```

### Run Protocols
To run protocols Inpect task, this could be run i





## TODOs

### Technical:
- [ ] add best-of-N 
- [ ] add refinement technique: self-reflection, self-critique
- [ ] decide on design of protocols: agents initialization inside or outside


### Experimental:
- [ ] Add self-awaraness for agents using prompt and dynamic label for messages