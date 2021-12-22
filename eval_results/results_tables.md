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

| Model/test set | AUC      | Acc    |
|----------------|----------|--------|
| Watson         | -------- | ------ |
| *normal*       | 0.904720 | 0.5334 |
| *adv*          | 0.158173 | 0.1975 |
| Polimi         | -------- | ------ |
| *normal*       |          |        |
| *adv*          | 0.792699 | 0.7003 |
| Sherlock       | -------- | ------ |
| *normal*       | 0.735761 | 0.5184 |
| *adv*          | 0.792630 | 0.5213 |

## Watson/Sherlock v2

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
