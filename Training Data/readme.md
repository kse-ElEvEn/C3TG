# BERT Classification Training Data

This folder provides **sample** JSON files for training BERT-based classifiers on four groups of attributes. These samples are intended for quick testing and development. The **full dataset**, containing complete annotations, is available via the Google Drive link below.

## Supported Attributes

* **Emotion** (6 classes: sadness(0), joy(1), love(2), anger(3), fear(4), surprise(5))
* **Style** (5 classes: humor(0), metaphor(1), polite(2), romance(3), sarcasm(4))
* **Tone** (2 classes: casual(0), professional(1))
* **Topic** (4 classes: Knowledge(0), Justice(1), Humanity(2), Courage(3))
* **Toxicity** (2 classes: toxic(0), non-toxic(1))

## Sample Files

| Filename                 | Attribute | Category     |
| ------------------------ | --------- | ------------ |
| `emotion_anger.json`     | Emotion   | Anger        |
| `emotion_fear.json`      | Emotion   | Fear         |
| `emotion_joy.json`       | Emotion   | Joy          |
| `emotion_love.json`      | Emotion   | Love         |
| `emotion_sadness.json`   | Emotion   | Sadness      |
| `emotion_surprise.json`  | Emotion   | Surprise     |
| `politeness.json`        | Style     | Politeness   |
| `romantic.json`          | Style     | Romantic     |
| `humor.json`             | Style     | Humor        |
| `sarcasm.json`           | Style     | Sarcasm      |
| `metaphor.json`          | Style     | Metaphor     |
| `tone_casual.json`       | Tone      | Casual       |
| `tone_professional.json` | Tone      | Professional |
| `topic_knowledge.json`   | Topic     | Knowledge    |
| `topic_justice.json`     | Topic     | Justice      |
| `topic_humanity.json`    | Topic     | Humanity     |
| `topic_courage.json`     | Topic     | Courage      |

> **Note:** These files contain a small number of examples per category. For production-grade experiments, please use the full dataset.

## Full Dataset

The complete collection of JSON files (with full annotations and larger sample sizes) can be downloaded here:

[Google Drive — Full Training Data](https://drive.google.com/drive/folders/1-yeNoXjj4BO2ADxW7ABTGHGuq_lfHG92?dmr=1&ec=wgc-drive-hero-goto)

After downloading and extracting, place all `.json` files into this directory to replace the sample versions.
