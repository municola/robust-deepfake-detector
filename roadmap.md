# Main Research Question

- Show that Polimi discriminator is not robust
- Present a reliable discriminator

# Workload

1. Get **normal** data set (256x256) to train/test our discriminator
	- Train:
		- real: FFHQ first 50k (+ CelebHQ ?)
		- fake: 50k using StyleGAN2 trained on FFHQ
	- Test:
		- real: FFHQ last 20k 
		- fake: 20k using StyleGAN3 trained on FFHQ
	- Data repo: [Polybox](https://polybox.ethz.ch/index.php/s/V3WwMQ3wnrW6rGN)
	
2. Write discriminator (PyTorch)
	- RestNet/EfficientNetv4/Inception/VGG + Layer (Fine-tuning)
	- Custom CNN + pooling
	
3. Evaluate our discriminator on the **normal** test set

4. Evaluate PoliMi discriminator on the **normal** test set
	- For reproducability/direct comparison
	- Most likely will have a better score than our discriminator
	
5. Generate an **adversarial** test set based on **normal** test set

6. Evaulate PoliMi discriminator on the **adversarial** test set
	- Hope for bad performance -> Key research question 😬
	- Here we see how we will write the storyline ("Better than you think", "Worse than you think")

7. Evaluate our discriminator on the **adversarial** test set
	- Also hope for bad performance otherwise adversarials don't work properly

8. Train our discriminator with adversarial training on the **normal** train set
	- Adv. training will iteratively incorporate adv. samples into training
	- These are correctly based on the **normal** train set
	- Try different attacks for training

9. Evaluate our discriminator (after adv. training) on the **adversarial** test set again
	- Hopefully improved performance as compared to before adv. training
	- Discriminator has become more robust 
	- Here we want to see that our ROC score is higher than PoliMi's score

10. Write final report (early deadline: 4. Jan)

# Timeline

- 15.Nov - 21.Nov : Kamm/Alex/Mo/Nici: Onboarding
- 22.Nov - 28.Nov : Kamm/Alex: Data sets (1); Mo/Nici: Discriminator (2)
- 29.Nov - 5.Dez  : Evaluation (3, 4); AdversarialGeneration (5)
- 6.Dez  - 12.Dez : 
- 13.Dez - 19.Dez : AgainEvaluation (6, 7); AdversarialTraining (8)
- 20.Dez - 26.Dez : AgainAgainEvaluation (9)
- Semester end
- 27.Dez - 2.Jan  : Writing Paper
- 2.Jan  - 4.Jan  : Buffer

# Concerns/Future Work

- Performance/Accuracy might also depend on image resolution. We expect to be better with higher resolution. Here we just do proof of concept.
- Performance of discriminator might depend on previously trained-on data sets, i.e. PoliMi discriminator will perform better than expected on **adversarial** test set (more robust than expected)
	






