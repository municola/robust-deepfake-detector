# Model evaluations

## Expectation

| Model/test set | AUC      | Acc    |
|----------------|----------|--------|
| Watson         | -------- | ------ |
| *normal*       |   High   |  High  |
| *adv* (1)      |   Low    |  Low   |
| Polimi         | -------- | ------ |
| *normal*       |   High   |  High  |
| *adv* (2)      |   Low    |  Low   |
| Sherlock       | -------- | ------ |
| *normal*       |   High   |  High  |
| *adv* (3)      |   High   |  High  |

(1) adversarial sample generation <br>
(2) adversarial sample transferability <br>
(3) adversarial training

## Watson/Sherlock v1
Initial model architecture (v1), adversarial attack FGSM with eps=0.05

| Model/test set | AUC      | Acc    |
|----------------|----------|--------|
| Watson         | -------- | ------ |
| *normal*       | 0.904720 | 0.5334 |
| *adv*          | 0.158173 | 0.1975 |
| Polimi         | -------- | ------ |
| *normal*       | 0.991053 | 0.7011 |
| *adv*          | 0.792699 | 0.7003 |
| Sherlock       | -------- | ------ |
| *normal*       | 0.735761 | 0.5184 |
| *adv*          | 0.792630 | 0.5213 |

## Watson/Sherlock v2
Initial model architecture (like v1), adversarial attack LinfPGD with eps=0.05

| Model/test set | AUC      | Acc    |
|----------------|----------|--------|
| Watson         | -------- | ------ |
| *normal*       | 0.904720 | 0.5334 |
| *adv*          | 0.078968 | 0.1890 |
| Polimi         | -------- | ------ |
| *normal*       | 0.991053 | 0.7011 |
| *adv*          | 0.716880 | 0.6211 |
| Sherlock       | -------- | ------ |
| *normal*       | 0.614204 | 0.4963 |
| *adv*          | 0.631619 | 0.5025 |

## Watson/Sherlock v3
Initial model architecture (like v1), adversarial attack LinfPGD with eps=0.01

| Model/test set | AUC      | Acc    |
|----------------|----------|--------|
| Watson         | -------- | ------ |
| *normal*       | 0.904720 | 0.5334 |
| *adv*          | 0.085589 | 0.1925 |
| Polimi         | -------- | ------ |
| *normal*       | 0.991053 | 0.7011 |
| *adv*          |          |        |
| Sherlock       | -------- | ------ |
| *normal*       | 0.857947 | 0.5131 |
| *adv*          | 0.841901 | 0.5115 |

## Watson/Sherlock v4

| Model/test set | AUC      | Acc    |
|----------------|----------|--------|
| Watson         | -------- | ------ |
| *normal*       |          |        |
| *adv*          |          |        |
| Polimi         | -------- | ------ |
| *normal*       |          |        |
| *adv*          |          |        |
| Sherlock       | -------- | ------ |
| *normal*       |          |        |
| *adv*          |          |        |
