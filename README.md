# Social Human Robot Embodied Conversation (SHREC) Dataset



<p align="center">
  <img src="https://github.com/dondongwon/website/blob/master/images/dong/SHREC.png?raw=true" width="300"/>
</p>


- **Authors**: Dong Won Lee, Yubin Kim, Sooyeon Jeong, Denison Guvenoz, Parker Malachowsky, Louis-Philippe Morency, Cynthia Breazeal, Hae Won Park  
- **Institutions**: MIT, Purdue University, Carnegie Mellon University  
- **License**: [Pending final approval]  


## 🧠 SHREC Dataset Summary
In full, the Social Human Robot Embodied Conversation (SHREC) Dataset is the first large-scale, real-world benchmark designed to evaluate **social reasoning** in **language** and **vision-language models** through physically embodied human-robot interactions (HRI). It contains:

- **~400 real-world interaction videos**
- **10,214 expert annotations**
- Labels for **social errors**, **competencies**, **rationales**, and **corrections**
- Coverage of **seven social attributes** critical for social intelligence

The dataset is split into 3 subsets:
- The **SHREC Wellness Home** subset contains real-world, longitudinal from [Jeong et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11094612/) recordings from an 8-week in-home study with adult participants aged 18–83. Participants engaged with a **socially assistive robot** designed to improve psychological well-being, affect, and readiness for change through evidence-based positive psychology interventions (PPIs).
- The **SHREC Wellness Dorm** subset contains longitudinal, **real-world human-robot interaction video data** data from [Jeong et al. (2020)](https://ieeexplore.ieee.org/document/9206085), where a **robotic positive psychology coach** was deployed in **MIT student dormitories**. Participants engaged in daily wellbeing sessions with the robot over the course of 1–4 weeks.
- The **SHREC Empathic** subset contains **real-world human-robot interaction video data** from [Shen et al. (2024)](https://aclanthology.org/2024.findings-acl.268.pdf), collected over a month-long deployment of social robots in participants’ homes, as participants engage in natural, empathic storytelling interactions with AI agents. 

It supports research in rapport-building, mental health intervention, and social reasoning in intimate, longitudinal HRI settings.

## 💾 Download Dataset from HuggingFace

To load this dataset into a pandas df: 

<pre>
import pandas as pd
from datasets import load_dataset
import pandas as pd
import glob
import os
from huggingface_hub import snapshot_download

repo_id = "MIT-personal-robots/shrec_wellness_home"

snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir="shrec_wellness_home", token = True)

parquet_files = glob.glob("downloaded_repo/**/*.parquet", recursive=True)

df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

</pre>



## 📦 Dataset Structure

Each interaction sample includes:

- `video_id`: Identifier for the interaction session
- `frame_paths`: List of image paths (15 selected frames from the video)
- `transcript`: Multi-turn dialogue between user and robot
- `label`: `"competence"`, `"error"`, or `"none"`
- `social_attributes`: List of relevant attributes from 7 core categories
- `rationale`: Explanation for the error or competence
- `correction`: Suggested repair if the segment is an error


## 🧪 Benchmark Tasks

The HSRI benchmark includes **8 tasks** spanning four core dimensions of social reasoning in human-robot interaction.

<p align="center">
  <img src="https://github.com/mitmedialab/SHREC/blob/main/images/task_figs_mar.pdf?raw=true" width="600"/>
</p>

### 1. Detecting Social Behavior

| Task                             | Description                                                              |
|----------------------------------|--------------------------------------------------------------------------|
| **Error / Competence / None Detection** | Classify the robot’s behavior as a social error, competence, or neither. |
| **Error Detection**              | Determine whether a given behavior constitutes a social error.           |



### 2. Identifying Social Attributes

| Task                             | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| **Social Attribute Identification** | Identify which of the seven social attributes are relevant to a given behavior. |
| **Multiple Attribute Detection** | Determine whether multiple social attributes are present in the behavior.   |

#### 7 Social Attributes

- **Emotions** – Identifying and responding to emotional expressions  
- **Engagement** – Monitoring user interest and presence  
- **Conversational Mechanics** – Managing turn-taking, timing, and pauses  
- **Knowledge State** – Tracking shared knowledge and references  
- **Intention** – Inferring the goals or motives behind actions  
- **Social Context & Relationships** – Acting appropriately based on context and social role  
- **Social Norms & Routines** – Following culturally appropriate social conventions  



### 3. Understanding Interaction Flow

| Task                    | Description                                                                          |
|-------------------------|--------------------------------------------------------------------------------------|
| **Pre-Condition Reasoning**  | Given the robot’s utterance, choose the plausible user behavior that came before.     |
| **Post-Condition Reasoning** | Given the user’s utterance, select the robot’s likely follow-up behavior.              |

These tasks are structured as **multiple-choice questions**, with distractors sampled from other real-world robot-user interactions to ensure contextual relevance.



### 4. Rationalizing & Correcting Social Errors

| Task                | Description                                                                          |
|---------------------|--------------------------------------------------------------------------------------|
| **Rationale Selection** | Choose the correct explanation for why the robot’s behavior was an error.            |
| **Correction Suggestion** | Select the most appropriate corrective action the robot should have taken instead. |

These tasks test a model's **diagnostic** (understanding what went wrong) and **prescriptive** (knowing how to fix it) social reasoning abilities.


## 🔍 Example Sample

```json
{
  "ID": "P15_s002-006",
  "sample_frame": "P15_s002-006/0000.png",
  "transcript": "AI Agent: (00:00:02) Hey there. How was your day today?\nUser A: (00:00:04) Good. How was yours?\n...\nAI Agent: (00:10:42) ... brighten our days.",
  "Annotations_A": [
    {
      "timestamp": {"start": 7.21, "end": 20.23},
      "error": true,
      "source": {"Verbal": true, "Non-Verbal": false},
      "attribute": {
        "Conversational Mechanics": true,
        "Intention": false,
        "Emotions": false,
        "Engagement": false,
        "Knowledge State": false,
        "Social Context &  Relationships",
        "Social Norms & Routines"
      },
      "rationale": "Delayed response and failure to understand participant.",
      "correction": "Should have responded within 2–3 seconds."
    }
  ],
  "Annotations_B": [
    {
      "..."
    }
  ],
  "Annotations_C": [
    {
      "..."
    }
  ],
  "framerate": 15.0,
  "frame_paths": [
    "P15_s002-006/0000.png",
    "P15_s002-006/0013.png",
    "P15_s002-006/0034.png",
    "..."
  ]
}

