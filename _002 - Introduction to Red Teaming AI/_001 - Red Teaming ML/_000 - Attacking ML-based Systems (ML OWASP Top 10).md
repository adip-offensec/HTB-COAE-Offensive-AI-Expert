Just like for [Web Applications](https://owasp.org/www-project-top-ten/), [Web APIs](https://owasp.org/www-project-api-security/), and [Mobile Applications](https://owasp.org/www-project-mobile-top-10/), OWASP has published a Top 10 list of security risks regarding the deployment and management of ML-based Systems, the [Top 10 for Machine Learning Security](https://owasp.org/www-project-machine-learning-security-top-10/). We will briefly discuss the ten risks to obtain an overview of security issues resulting from ML-based systems.

|ID|Description|
|---|---|
|ML01|`Input Manipulation Attack`: Attackers modify input data to cause incorrect or malicious model outputs.|
|ML02|`Data Poisoning Attack`: Attackers inject malicious or misleading data into training data, compromising model performance or creating backdoors.|
|ML03|`Model Inversion Attack`: Attackers train a separate model to reconstruct inputs from model outputs, potentially revealing sensitive information.|
|ML04|`Membership Inference Attack`: Attackers analyze model behavior to determine whether data was included in the model's training data set, potentially revealing sensitive information.|
|ML05|`Model Theft`: Attackers train a separate model from interactions with the original model, thereby stealing intellectual property.|
|ML06|`AI Supply Chain Attacks`: Attackers exploit vulnerabilities in any part of the ML supply chain.|
|ML07|`Transfer Learning Attack`: Attackers manipulate the baseline model that is subsequently fine-tuned by a third-party. This can lead to biased or backdoored models.|
|ML08|`Model Skewing`: Attackers skew the model's behavior for malicious purposes, for instance, by manipulating the training data set.|
|ML09|`Output Integrity Attack`: Attackers manipulate a model's output before processing, making it look like the model produced a different output.|
|ML10|`Model Poisoning`: Attackers manipulate the model's weights, compromising model performance or creating backdoors.|

---

## Input Manipulation Attack (ML01)

As the name suggests, input manipulation attacks comprise any type of attack against an ML model that results from manipulating the input data. Typically, the result of these attacks is unexpected behavior of the ML model that deviates from the intended behavior. The impact depends highly on the concrete scenario and circumstances in which the model is used. It can range from financial and reputational damage to legal consequences or data loss.

Many real-world input manipulation attack vectors involve applying small perturbations to benign input data, resulting in unexpected behavior from the ML model. In contrast, the perturbations are so small that the input looks benign to the human eye. For instance, consider a self-driving car that utilizes an ML-based system for image classification of road signs to detect the current speed limit, stop signs, and other relevant information. In an input manipulation attack, an attacker could add small perturbations, such as particularly placed dirt specks, small stickers, or graffiti, to road signs. While these perturbations appear harmless to the human eye, they could lead to the misclassification of the sign by the ML-based system. This can have deadly consequences for passengers of the vehicle. For more details on this attack vector, check out [this](https://arxiv.org/pdf/1707.08945) and [this](https://arxiv.org/pdf/2307.08278) paper.

![Diagram showing Training Phase with Training Data leading to a Learned Model, and Test Phase with ML-based Service affected by Adversarial Examples causing misclassification.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/294/diagram_3.png)

---

## Data Poisoning Attack (ML02)

Data poisoning attacks on ML-based systems involve injecting malicious or misleading data into the training data set to compromise the model's accuracy, performance, or behavior. As discussed before, the quality of any ML model is highly dependent on the quality of the training data. As such, these attacks can cause a model to make incorrect predictions, misclassify certain inputs, or behave unpredictably in specific scenarios. ML models often rely on large-scale, automated data collection from various sources, making them more susceptible to tampering, especially when the sources are unverified or sourced from public domains.

As an example, assume an adversary is able to inject malicious data into the training data set for a model used in antivirus software to determine whether a given binary is malware. The adversary may manipulate the training data to effectively establish a backdoor, enabling them to create custom malware that the model classifies as benign. More details about installing backdoors through data poisoning attacks are discussed in [this](https://arxiv.org/pdf/2408.13221) paper.

---

## Model Inversion Attack (ML03)

In model inversion attacks, an adversary trains a separate ML model on the output of the target model to reconstruct information about the target model's inputs. Since the model trained by the adversary operates on the target model's output and reconstructs information about the inputs, it `inverts` the target model's functionality, hence the name `model inversion attack`.

These attacks are particularly impactful if the input data contains sensitive information—for instance, models processing medical data, such as classifiers used in cancer detection. If an inverse model can reconstruct information about a patient's medical information based on the classifier's output, sensitive information is at risk of being leaked to the adversary. Furthermore, model inversion attacks are more challenging to execute if the target model provides less output information. For instance, successfully training an inverse model becomes much more challenging if a classification model only outputs the target class instead of every output probability.

An approach for model inversion of language models is discussed in [this](https://arxiv.org/pdf/2311.13647) paper.

---

## Membership Inference Attack (ML04)

Membership inference attacks aim to determine whether a specific data sample was included in the model's original training data set. By carefully analyzing the model's responses to different inputs, an attacker can infer which data points the model "remembers" from the training process. If a model is trained on sensitive data such as medical or financial information, this can pose serious privacy issues. This attack is especially concerning in publicly accessible or shared models, such as those in cloud-based or machine learning-as-a-service (MLaaS) environments. The success of membership inference attacks often hinges on the differences in the model's behavior when handling training versus non-training data, as models typically exhibit higher confidence or lower prediction error on samples they have seen before.

An extensive assessment of the performance of membership inference attacks on language models is performed in [this](https://arxiv.org/pdf/2402.07841) paper.

![Diagram showing Synthetic Data and Source Data used by a Generative Model, leading to an Inference Model. Adversary exploits this, risking patient data disclosure.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/294/diagram_6.png)

---

## Model Theft (ML05)

Model theft or model extraction attacks aim to duplicate or approximate the functionality of a target model without direct access to its underlying architecture or parameters. In these attacks, an adversary interacts with an ML model and systematically queries it to gather sufficient data about its decision-making behavior to duplicate the model. By observing sufficient outputs for various inputs, attackers can train their own replica model with a similar performance.

Model theft threatens the intellectual property of organizations investing in proprietary ML models, potentially resulting in financial or reputational damage. Furthermore, model theft may expose sensitive insights embedded within the model, such as learned patterns from sensitive training data.

For more details on the effectiveness of model theft attacks on a specific type of neural network, check out [this](https://arxiv.org/pdf/2305.13584) paper.

![Diagram showing an attacker using a theft data set to train a stolen model via queries to an MLaaS platform.](https://cdn.services-k8s.prod.aws.htb.systems/content/modules/294/diagram_4.png)

---

## AI Supply Chain Attacks (ML06)

Supply chain attacks on ML-based systems target the complex, interconnected ecosystem involved in creating, deploying, and maintaining ML models. These attacks exploit vulnerabilities in any part of the ML pipeline, such as third-party data sources, libraries, or pre-trained models, to compromise the model's integrity, security, or performance. The supply chain of ML-based systems consists of more components than traditional IT systems, due to their reliance on large amounts of data. Details of supply chain attacks, including their impact, depend highly on the specific vulnerability exploited. For instance, they can result in manipulated models that perform differently than intended. The risk of supply chain attacks has grown as ML systems increasingly rely on open-source tools, publicly available data sets, and pre-trained models from external sources.

For more general information about supply chain attacks, check out the [Supply Chain Attacks](https://academy.hackthebox.com/module/details/243) module.

---

## Transfer Learning Attack (ML07)

Open-source pre-trained models are used as a baseline for many ML model deployments due to the high computational cost of training models from scratch. New models are then built on top of these pre-trained models by applying additional training to fine-tune the model to the specific task it is supposed to execute. In transfer learning attacks, adversaries exploit this transfer process by manipulating the pre-trained model. Security issues such as backdoors or biases may arise if these manipulations persist in the fine-tuned model. Even if the data set used for fine-tuning is benign, malicious behavior from the pre-trained model may carry over to the final ML-based system.

---

## Model Skewing (ML08)

In model skewing attacks, an adversary attempts to deliberately skew a model's output in a biased manner that favors the adversary's objectives. They can achieve this by injecting biased, misleading, or incorrect data into the training data set to influence the model's output toward maliciously biased outcomes.

For instance, assume our previously discussed scenario of an ML model that classifies whether a given binary is malware. An adversary might be able to skew the model to classify malware as benign binaries by including incorrectly labeled training data into the training data set. In particular, an attacker might add their own malware binary with a `benign` label to the training data to evade detection by the trained model.

---

## Output Integrity Attack (ML09)

If an attacker can alter the output produced by an ML-based system, they can execute an output integrity attack. This attack does not target the model itself, but only its output. More specifically, the attacker does not manipulate the model directly; instead, they intercept the model's output before the respective target entity processes it. They manipulate the output to make it appear as though the model has produced a different result. Detecting output integrity attacks is challenging because the model often appears to function normally upon inspection, rendering traditional model-based security measures insufficient.

As an example, consider the ML malware classifier again. Let us assume that the system acts based on the classifier's result and deletes all binaries from the disk if classified as malware. If an attacker can manipulate the classifier's output before the succeeding system acts, they can introduce malware by exploiting an output integrity attack. After copying their malware to the target system, the classifier will classify the binary as malicious. The attacker then manipulates the model's output to the label `benign` instead of `malicious`. Subsequently, the succeeding system does not delete the malware as it assumes the binary was not classified as malware.

---

## Model Poisoning (ML10)

While data poisoning attacks target the model's training data and, thus, indirectly, the model's parameters, model poisoning attacks target the model's parameters directly. As such, an adversary needs access to the model parameters to execute this type of attack. Furthermore, manipulating the parameters in a targeted malicious way can be challenging. While changing model parameters arbitrarily will most certainly result in lower model performance, getting the model to deviate from its intended behavior in a deliberate way requires well-thought-out and nuanced parameter manipulations. The impact of model poisoning attacks is similar to data poisoning attacks, as it can lead to incorrect predictions, misclassification of certain inputs, or unpredictable behavior in specific scenarios.

For more details regarding an actual model poisoning attack vector, check out [this](https://arxiv.org/pdf/2405.20975) paper.

