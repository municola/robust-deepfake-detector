# Model evaluations

| Model    | test set | AUC      | Acc    |
|----------|----------|----------|--------|
| Polimi   |  normal  | 0.991053 | 0.7011 |
|          |   adv1   | 0.792699 | 0.7003 |
|          |   adv2   | 0.716880 | 0.6211 |
|          |   adv3   | 0.931072 | 0.7169 |
|          |          |          |        |
| Watson   |  normal  | 0.992820 | 0.9464 |
|          |   adv1   | 0.981916 | 0.9182 |
|          |   adv2   | 0.898157 | 0.7183 |
|          |   adv3   | 0.988455 | 0.9394 |
|          |          |          |        |
| Sherlock |  normal  | 0.997113 | 0.9688 |
|          |   adv1   | 0.996207 | 0.9696 |
|          |   adv2   | 0.980525 | 0.8475 |
|          |   adv3   | 0.996076 | 0.9608 |
|          |          |          |        |
| Moriarty |  normal  | 0.904720 | 0.5334 |
|          |   adv1   | 0.158173 | 0.1975 |
|          |   adv2   | 0.078968 | 0.1890 |
|          |   adv3   | 0.085589 | 0.1925 |

normal: normal testset with stylegan3 and ffhq <br>
adv1: adversarial attack FGSM with eps=0.01 <br>
adv2: adversarial attack LinfPGD with eps=0.05 <br>
adv3: adversarial attack LinfPGD with eps=0.01 <br>
